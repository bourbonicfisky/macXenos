/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2020 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#include "xenia/gpu/trace_reader.h"

#include <cstring>

#include "third_party/snappy/snappy.h"
#include "xenia/base/filesystem.h"
#include "xenia/base/logging.h"
#include "xenia/base/mapped_memory.h"
#include "xenia/base/math.h"
#include "xenia/base/platform.h"
#include "xenia/base/xxhash.h"
#include "xenia/gpu/packet_disassembler.h"
#include "xenia/gpu/trace_protocol.h"
#include "xenia/memory.h"

namespace xe {
namespace gpu {

bool TraceReader::Open(const std::string_view path) {
  Close();

  mmap_.reset();
#if XE_PLATFORM_ANDROID
  if (xe::filesystem::IsAndroidContentUri(path)) {
    mmap_ =
        MappedMemory::OpenForAndroidContentUri(path, MappedMemory::Mode::kRead);
  }
#endif  // XE_PLATFORM_ANDROID
  if (!mmap_) {
    mmap_ = MappedMemory::Open(xe::to_path(path), MappedMemory::Mode::kRead);
  }
  if (!mmap_) {
    return false;
  }

  trace_data_ = reinterpret_cast<const uint8_t*>(mmap_->data());
  size_t mapped_size = mmap_->size();
  if (mapped_size < sizeof(TraceHeader)) {
    XELOGE("Trace file is too small to contain a header");
    Close();
    return false;
  }
  trace_size_ = mapped_size;

  // A v2 bundle starts with an exact v1 trace and ends with its section table
  // and footer. Detect it from the footer rather than changing the command
  // stream header, preserving byte identity with the legacy sibling.
  bool bundle_extension =
      path.size() >= 5 && path.substr(path.size() - 5) == ".xtr2";
  bool bundle_found = false;
  if (mapped_size >= sizeof(TraceHeader) + sizeof(TraceBundleFooter)) {
    TraceBundleFooter footer = {};
    size_t footer_offset = mapped_size - sizeof(footer);
    std::memcpy(&footer, trace_data_ + footer_offset, sizeof(footer));
    if (footer.magic == kTraceBundleMagic) {
      bundle_found = true;
      if (footer.version != kTraceBundleFormatVersion ||
          footer.footer_size != sizeof(TraceBundleFooter) ||
          footer.legacy_trace_size < sizeof(TraceHeader) ||
          footer.legacy_trace_size > footer_offset ||
          footer.section_table_offset < footer.legacy_trace_size ||
          footer.section_table_offset > footer_offset ||
          footer.section_count >
              (footer_offset - footer.section_table_offset) /
                  sizeof(TraceBundleSection) ||
          footer.section_table_offset +
                  uint64_t(footer.section_count) * sizeof(TraceBundleSection) !=
              footer_offset) {
        XELOGE("Portable trace bundle footer is invalid");
        Close();
        return false;
      }

      uint64_t legacy_hash =
          XXH3_64bits(trace_data_, size_t(footer.legacy_trace_size));
      if (legacy_hash != footer.legacy_trace_xxh3_64) {
        XELOGE(
            "Portable trace bundle legacy payload hash mismatch (expected "
            "{:016X}, got {:016X})",
            footer.legacy_trace_xxh3_64, legacy_hash);
        Close();
        return false;
      }

      for (uint32_t i = 0; i < footer.section_count; ++i) {
        TraceBundleSection section = {};
        std::memcpy(&section,
                    trace_data_ + footer.section_table_offset +
                        size_t(i) * sizeof(section),
                    sizeof(section));
        if (section.offset < footer.legacy_trace_size ||
            section.offset > footer.section_table_offset ||
            section.length > footer.section_table_offset - section.offset) {
          XELOGE("Portable trace bundle section {} is out of bounds", i);
          Close();
          return false;
        }
        uint64_t section_hash =
            XXH3_64bits(trace_data_ + section.offset, size_t(section.length));
        if (section_hash != section.xxh3_64) {
          XELOGE("Portable trace bundle section {} hash mismatch", i);
          Close();
          return false;
        }
        if (section.type == TraceBundleSectionType::kMetadataJson) {
          metadata_json_ = std::string_view(
              reinterpret_cast<const char*>(trace_data_ + section.offset),
              size_t(section.length));
        }
      }
      if (metadata_json_.empty()) {
        XELOGE("Portable trace bundle has no metadata JSON section");
        Close();
        return false;
      }
      bundle_format_version_ = footer.version;
      trace_size_ = size_t(footer.legacy_trace_size);
    }
  }
  if (bundle_extension && !bundle_found) {
    XELOGE("Portable trace bundle footer is missing");
    Close();
    return false;
  }

  // Verify version.
  auto header = reinterpret_cast<const TraceHeader*>(trace_data_);
  if (header->version != kTraceFormatVersion) {
    XELOGE("Trace format version mismatch, code has {}, file has {}",
           kTraceFormatVersion, header->version);
    if (header->version < kTraceFormatVersion) {
      XELOGE("You need to regenerate your trace for the latest version");
    }
    return false;
  }

  XELOGI("Mapped {}b trace from {}", trace_size_, path);
  if (bundle_format_version_) {
    XELOGI("    Bundle: v{} ({}b metadata)", bundle_format_version_,
           metadata_json_.size());
  }
  XELOGI("   Version: {}", header->version);
  auto commit_str = std::string(header->build_commit_sha,
                                xe::countof(header->build_commit_sha));
  XELOGI("    Commit: {}", commit_str);
  XELOGI("  Title ID: {}", header->title_id);

  ParseTrace();

  return true;
}

void TraceReader::Close() {
  frames_.clear();
  mmap_.reset();
  trace_data_ = nullptr;
  trace_size_ = 0;
  bundle_format_version_ = 0;
  metadata_json_ = {};
}

void TraceReader::ParseTrace() {
  // Skip file header.
  auto trace_ptr = trace_data_;
  trace_ptr += sizeof(TraceHeader);

  Frame current_frame;
  current_frame.start_ptr = trace_ptr;
  const PacketStartCommand* packet_start = nullptr;
  const uint8_t* packet_start_ptr = nullptr;
  const uint8_t* last_ptr = trace_ptr;
  bool pending_break = false;
  auto current_command_buffer = new CommandBuffer();
  current_frame.command_tree =
      std::unique_ptr<CommandBuffer>(current_command_buffer);

  while (trace_ptr < trace_data_ + trace_size_) {
    ++current_frame.command_count;
    auto type = static_cast<TraceCommandType>(xe::load<uint32_t>(trace_ptr));
    switch (type) {
      case TraceCommandType::kPrimaryBufferStart: {
        auto cmd =
            reinterpret_cast<const PrimaryBufferStartCommand*>(trace_ptr);
        trace_ptr += sizeof(*cmd) + cmd->count * 4;
        break;
      }
      case TraceCommandType::kPrimaryBufferEnd: {
        auto cmd = reinterpret_cast<const PrimaryBufferEndCommand*>(trace_ptr);
        trace_ptr += sizeof(*cmd);
        break;
      }
      case TraceCommandType::kIndirectBufferStart: {
        auto cmd =
            reinterpret_cast<const IndirectBufferStartCommand*>(trace_ptr);
        trace_ptr += sizeof(*cmd) + cmd->count * 4;

        // Traverse down a level.
        auto sub_command_buffer = new CommandBuffer();
        sub_command_buffer->parent = current_command_buffer;
        current_command_buffer->commands.push_back(
            CommandBuffer::Command(sub_command_buffer));
        current_command_buffer = sub_command_buffer;
        break;
      }
      case TraceCommandType::kIndirectBufferEnd: {
        auto cmd = reinterpret_cast<const IndirectBufferEndCommand*>(trace_ptr);
        trace_ptr += sizeof(*cmd);

        // IB packet is wrapped in a kPacketStart/kPacketEnd. Skip the end.
        auto end_cmd = reinterpret_cast<const PacketEndCommand*>(trace_ptr);
        assert_true(end_cmd->type == TraceCommandType::kPacketEnd);
        trace_ptr += sizeof(*cmd);

        // Go back up a level. If parent is null, this frame started in an
        // indirect buffer.
        if (current_command_buffer->parent) {
          current_command_buffer = current_command_buffer->parent;
        }
        break;
      }
      case TraceCommandType::kPacketStart: {
        auto cmd = reinterpret_cast<const PacketStartCommand*>(trace_ptr);
        packet_start_ptr = trace_ptr;
        packet_start = cmd;
        trace_ptr += sizeof(*cmd) + cmd->count * 4;
        break;
      }
      case TraceCommandType::kPacketEnd: {
        auto cmd = reinterpret_cast<const PacketEndCommand*>(trace_ptr);
        trace_ptr += sizeof(*cmd);
        if (!packet_start_ptr) {
          continue;
        }
        auto packet_category = PacketDisassembler::GetPacketCategory(
            packet_start_ptr + sizeof(*packet_start));
        switch (packet_category) {
          case PacketCategory::kDraw: {
            Frame::Command command;
            command.type = Frame::Command::Type::kDraw;
            command.head_ptr = packet_start_ptr;
            command.start_ptr = last_ptr;
            command.end_ptr = trace_ptr;
            current_frame.commands.push_back(std::move(command));
            last_ptr = trace_ptr;
            current_command_buffer->commands.push_back(CommandBuffer::Command(
                uint32_t(current_frame.commands.size() - 1)));
            break;
          }
          case PacketCategory::kSwap: {
            Frame::Command command;
            command.type = Frame::Command::Type::kSwap;
            command.head_ptr = packet_start_ptr;
            command.start_ptr = last_ptr;
            command.end_ptr = trace_ptr;
            current_frame.commands.push_back(std::move(command));
            last_ptr = trace_ptr;
            current_command_buffer->commands.push_back(CommandBuffer::Command(
                uint32_t(current_frame.commands.size() - 1)));
          } break;
          case PacketCategory::kGeneric: {
            // Ignored.
            break;
          }
        }
        if (pending_break) {
          current_frame.end_ptr = trace_ptr;
          frames_.push_back(std::move(current_frame));
          current_command_buffer = new CommandBuffer();
          current_frame.command_tree =
              std::unique_ptr<CommandBuffer>(current_command_buffer);
          current_frame.start_ptr = trace_ptr;
          current_frame.end_ptr = nullptr;
          current_frame.command_count = 0;
          pending_break = false;
        }
        break;
      }
      case TraceCommandType::kMemoryRead: {
        auto cmd = reinterpret_cast<const MemoryCommand*>(trace_ptr);
        trace_ptr += sizeof(*cmd) + cmd->encoded_length;
        break;
      }
      case TraceCommandType::kMemoryWrite: {
        auto cmd = reinterpret_cast<const MemoryCommand*>(trace_ptr);
        trace_ptr += sizeof(*cmd) + cmd->encoded_length;
        break;
      }
      case TraceCommandType::kEdramSnapshot: {
        auto cmd = reinterpret_cast<const EdramSnapshotCommand*>(trace_ptr);
        trace_ptr += sizeof(*cmd) + cmd->encoded_length;
        break;
      }
      case TraceCommandType::kEvent: {
        auto cmd = reinterpret_cast<const EventCommand*>(trace_ptr);
        trace_ptr += sizeof(*cmd);
        switch (cmd->event_type) {
          case EventCommand::Type::kSwap: {
            pending_break = true;
            break;
          }
        }
        break;
      }
      case TraceCommandType::kRegisters: {
        auto cmd = reinterpret_cast<const RegistersCommand*>(trace_ptr);
        trace_ptr += sizeof(*cmd) + cmd->encoded_length;
        break;
      }
      case TraceCommandType::kGammaRamp: {
        auto cmd = reinterpret_cast<const GammaRampCommand*>(trace_ptr);
        trace_ptr += sizeof(*cmd) + cmd->encoded_length;
        break;
      }
      default:
        // Broken trace file?
        assert_unhandled_case(type);
        break;
    }
  }
  if (pending_break || current_frame.command_count) {
    current_frame.end_ptr = trace_ptr;
    frames_.push_back(std::move(current_frame));
  }
}

bool TraceReader::DecompressMemory(MemoryEncodingFormat encoding_format,
                                   const void* src, size_t src_size, void* dest,
                                   size_t dest_size) {
  switch (encoding_format) {
    case MemoryEncodingFormat::kNone:
      assert_true(src_size == dest_size);
      std::memcpy(dest, src, src_size);
      return true;
    case MemoryEncodingFormat::kSnappy:
      return snappy::RawUncompress(reinterpret_cast<const char*>(src), src_size,
                                   reinterpret_cast<char*>(dest));
    default:
      assert_unhandled_case(encoding_format);
      return false;
  }
}

}  // namespace gpu
}  // namespace xe
