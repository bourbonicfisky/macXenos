#pragma once

#if !defined(NDEBUG)
#define XE_ENABLE_PM4_DISASM 1
#endif

using namespace xe::gpu::xenos;
void COMMAND_PROCESSOR::ExecuteIndirectBuffer(uint32_t ptr,
                                              uint32_t count) XE_RESTRICT {
  SCOPE_profile_cpu_f("gpu");

  trace_writer_.WriteIndirectBufferStart(ptr, count * sizeof(uint32_t));
  if (count != 0) {
    RingBuffer old_reader = reader_;

    // Execute commands!
    new (&reader_)
        RingBuffer(memory_->TranslatePhysical(ptr), count * sizeof(uint32_t));
    reader_.set_write_offset(count * sizeof(uint32_t));
    // prefetch the wraparound range
    // it likely is already in L3 cache, but in a zen system it may be another
    // chiplets l3
    reader_.BeginPrefetchedRead<swcache::PrefetchTag::Level2>(
        COMMAND_PROCESSOR::GetCurrentRingReadCount());
    do {
      if (COMMAND_PROCESSOR::ExecutePacket()) {
        continue;
      } else {
        // Return up a level if we encounter a bad packet.
        XELOGE("**** INDIRECT RINGBUFFER: Failed to execute packet.");
        assert_always();
        break;
      }
    } while (reader_.read_count());

    trace_writer_.WriteIndirectBufferEnd();
    reader_ = old_reader;
  } else {
    // rare, but i've seen it happen! (and then a division by 0 occurs)
    return;
  }
}
XE_NOINLINE
static void LOGU32s(logging::LoggerBatch<LogLevel::Debug>& logger,
                    const std::vector<uint32_t>& values) {
  bool first = true;
#if 0
  XELOGD("[ ");
  for (auto&& val : values) {
    if (first) {
      XELOGD("0x{:08X}", val);
      first = false;
    } else {
      XELOGD(", 0x{:08X}", val);
    }
  }
  XELOGD(" ]");
#else

  for (auto&& val : values) {
    if (first) {
      logger("0x{:08X}", val);
      first = false;
    } else {
      logger(", 0x{:08X}", val);
    }
  }
#endif
}

std::string GenerateRegnameForPm4Print(uint32_t reg) {
  auto reg_info = RegisterFile::GetRegisterInfo(reg);

  if (reg_info) {
    return reg_info->name;
  } else {
    return fmt::format("Unknown_Reg_{:04X}", reg);
  }
}

XE_NOINLINE
void COMMAND_PROCESSOR::DisassembleCurrentPacket() XE_RESTRICT {
  xe::gpu::PacketInfo packet_info;

  logging::LoggerBatch<LogLevel::Debug> logger{};

  if (PacketDisassembler::DisasmPacket(reader_.buffer() + reader_.read_offset(),
                                       &packet_info)) {
    logger("CP - {}, count {}, predicated = {}\n", packet_info.type_info->name,
           packet_info.count, packet_info.predicated);

#define LOG_ACTION_FIELD(__type, name) \
  logger("\t" #name " = {:08X}\n", static_cast<uint32_t>(action.__type.name))
#define LOG_ACTION_FIELD_DEC(__type, name) \
  logger("\t" #name " = {}\n", static_cast<size_t>(action.__type.name))

#define LOG_ENDIANNESS(__type, name) \
  logger("\t" #name " = {}\n",       \
         xenos::GetEndianEnglishDescription(action.__type.name))
#define LOG_PRIMTYPE(__type, name) \
  logger("\t" #name " = {}\n",     \
         xenos::GetPrimitiveTypeEnglishDescription(action.__type.name))
    for (auto&& action : packet_info.actions) {
      using PType = PacketAction::Type;
      switch (action.type) {
        case PType::kRegisterWrite:
          break;
        case PType::kSetBinMask:
          logger("\tSetBinMask {}\n", action.set_bin_mask.value);
          break;
        case PType::kSetBinSelect:
          logger("\tSetBinSelect {}\n", action.set_bin_select.value);
          break;
        case PType::kMeInit:
          logger("\tMeInit - ");
          LOGU32s(logger, action.words);
          logger("\n");
          break;
        case PType::kGenInterrupt:
          logger("\tGenInterrupt for cpu mask {:04X}\n",
                 action.gen_interrupt.cpu_mask);
          break;
        case PType::kSetBinMaskHi:
        case PType::kSetBinMaskLo:
        case PType::kSetBinSelectHi:
        case PType::kSetBinSelectLo:
          LOG_ACTION_FIELD(lohi_op, value);
          break;
        case PType::kWaitRegMem:
          LOG_ACTION_FIELD(wait_reg_mem, wait_info);
          LOG_ACTION_FIELD(wait_reg_mem, poll_reg_addr);
          LOG_ACTION_FIELD(wait_reg_mem, ref);
          LOG_ACTION_FIELD(wait_reg_mem, mask);
          LOG_ACTION_FIELD(wait_reg_mem, wait);
          break;
        case PType::kRegRmw: {
          uint32_t rmw_info = action.reg_rmw.rmw_info;
          uint32_t and_mask = action.reg_rmw.and_mask;
          uint32_t or_mask = action.reg_rmw.or_mask;

          uint32_t and_mask_is_reg = (rmw_info >> 31) & 0x1;
          uint32_t or_mask_is_reg = (rmw_info >> 30) & 0x1;

          std::string and_mask_str;

          if (and_mask_is_reg) {
            and_mask_str = GenerateRegnameForPm4Print(and_mask & 0x1FFF);
          } else {
            and_mask_str = fmt::format("0x{:08X}", and_mask);
          }
          std::string or_mask_str;

          if (or_mask_is_reg) {
            or_mask_str = GenerateRegnameForPm4Print(or_mask & 0x1FFF);
          } else {
            or_mask_str = fmt::format("0x{:08X}", or_mask);
          }

          std::string dest = GenerateRegnameForPm4Print(rmw_info & 0x1FFF);

          logger("\t{} = ({} & {}) | {}\n", dest, dest, and_mask_str,
                 or_mask_str);
          LOG_ACTION_FIELD(reg_rmw, rmw_info);
          LOG_ACTION_FIELD(reg_rmw, and_mask);
          LOG_ACTION_FIELD(reg_rmw, or_mask);

          break;
        }
        case PType::kCondWrite:
          LOG_ACTION_FIELD(cond_write, wait_info);
          LOG_ACTION_FIELD(cond_write, poll_reg_addr);
          LOG_ACTION_FIELD(cond_write, ref);
          LOG_ACTION_FIELD(cond_write, mask);
          LOG_ACTION_FIELD(cond_write, write_reg_addr);
          LOG_ACTION_FIELD(cond_write, write_data);
          break;

        case PType::kEventWrite:
          LOG_ACTION_FIELD(event_write, initiator);
          break;
        case PType::kEventWriteSHD:
          LOG_ACTION_FIELD(event_write_shd, initiator);
          LOG_ACTION_FIELD(event_write_shd, address);
          LOG_ACTION_FIELD(event_write_shd, value);
          break;
        case PType::kEventWriteExt:
          LOG_ACTION_FIELD(event_write_ext, unk0);
          LOG_ACTION_FIELD(event_write_ext, unk1);
          break;
        case PType::kDrawIndx:
          LOG_ACTION_FIELD(draw_indx, dword0);
          LOG_ACTION_FIELD(draw_indx, dword1);
          LOG_ACTION_FIELD_DEC(draw_indx, index_count);
          LOG_PRIMTYPE(draw_indx, prim_type);
          LOG_ACTION_FIELD(draw_indx, src_sel);
          LOG_ACTION_FIELD(draw_indx, guest_base);
          LOG_ACTION_FIELD_DEC(draw_indx, index_size);
          LOG_ENDIANNESS(draw_indx, endianness);
          break;
        case PType::kDrawIndx2:
          LOG_ACTION_FIELD(draw_indx2, dword0);
          LOG_ACTION_FIELD_DEC(draw_indx2, index_count);
          LOG_PRIMTYPE(draw_indx2, prim_type);
          LOG_ACTION_FIELD(draw_indx2, src_sel);
          LOG_ACTION_FIELD_DEC(draw_indx2, indices_size);
          logger("Indices = ");
          LOGU32s(logger, action.words);
          logger("\n");
          break;
        case PType::kInvalidateState:
          LOG_ACTION_FIELD(invalidate_state, state_mask);
          break;
        case PType::kImLoad:
          LOG_ACTION_FIELD(im_load, shader_type);
          LOG_ACTION_FIELD(im_load, addr);
          LOG_ACTION_FIELD(im_load, start);
          LOG_ACTION_FIELD_DEC(im_load, size_dwords);
          break;
        case PType::kImLoadImmediate:
          LOG_ACTION_FIELD_DEC(im_load_imm, shader_type);
          LOG_ACTION_FIELD(im_load_imm, start);
          LOG_ACTION_FIELD_DEC(im_load_imm, size_dwords);
          logger("Shader instruction words = ");
          LOGU32s(logger, action.words);
          logger("\n");
          break;
        case PType::kWaitForIdle:
          LOG_ACTION_FIELD(wait_for_idle, probably_unused);
          break;
        case PType::kContextUpdate:
          LOG_ACTION_FIELD(context_update, maybe_unused);
          break;
        case PType::kVizQuery:
          LOG_ACTION_FIELD_DEC(vizquery, id);
          LOG_ACTION_FIELD_DEC(vizquery, end);
          LOG_ACTION_FIELD(vizquery, dword0);
          break;
        case PType::kEventWriteZPD:
          LOG_ACTION_FIELD(event_write_zpd, initiator);

          break;
        case PType::kMemWrite:
          LOG_ACTION_FIELD(mem_write, addr);
          LOG_ENDIANNESS(mem_write, endianness);
          logger("Values to write (with GpuSwap pre-applied) = ");
          LOGU32s(logger, action.words);
          logger("\n");
          break;
        case PType::kRegToMem:
          logger("{}\n", GenerateRegnameForPm4Print(action.reg2mem.reg_addr));
          LOG_ACTION_FIELD(reg2mem, mem_addr);
          LOG_ENDIANNESS(reg2mem, endianness);
          break;
        case PType::kIndirBuffer:
          LOG_ACTION_FIELD(indir_buffer, list_ptr);
          LOG_ACTION_FIELD_DEC(indir_buffer, list_length);
          break;
        case PType::kXeSwap:
          LOG_ACTION_FIELD(xe_swap, frontbuffer_ptr);
          break;
      }
    }
  } else {
    logger("Unknown packet! Failed to disassemble.\n");
  }
  logger.submit('d');
}
bool COMMAND_PROCESSOR::ExecutePacket() {
#if XE_ENABLE_PM4_DISASM == 1
  if (cvars::disassemble_pm4 && logging::ShouldLog(LogLevel::Debug)) {
    COMMAND_PROCESSOR::DisassembleCurrentPacket();
  }
#endif
  const uint32_t packet = reader_.ReadAndSwap<uint32_t>();
  const uint32_t packet_type = packet >> 30;

  XE_LIKELY_IF(packet && packet != 0x0BADF00D) {
    XE_LIKELY_IF((packet != 0xCDCDCDCD)) {
    actually_execute_packet:
      // chrispy: reorder checks by probability
      XE_LIKELY_IF(packet_type == 3) {
        return COMMAND_PROCESSOR::ExecutePacketType3(packet);
      }
      else {
        if (packet_type ==
            0) {  // dont know whether 0 or 1 are the next most frequent
          return COMMAND_PROCESSOR::ExecutePacketType0(packet);
        } else {
          if (packet_type == 1) {
            return COMMAND_PROCESSOR::ExecutePacketType1(packet);
          } else {
            // originally there was a default case that msvc couldn't optimize
            // away because it doesnt have value range analysis but in reality
            // there is no default, a uint32_t >> 30 only has 4 possible values
            // and all are covered here
            // return COMMAND_PROCESSOR::ExecutePacketType2(packet);
            // executepackettype2 is identical
            goto handle_bad_packet;
          }
        }
      }
    }
    else {
      XELOGW("GPU packet is CDCDCDCD - probably read uninitialized memory!");
      goto actually_execute_packet;
    }
  }
  else {
  handle_bad_packet:
    trace_writer_.WritePacketStart(uint32_t(reader_.read_ptr() - 4), 1);
    trace_writer_.WritePacketEnd();
    return true;
  }
}
XE_NOINLINE
XE_COLD
bool COMMAND_PROCESSOR::ExecutePacketType0_CountOverflow(uint32_t count) {
  XELOGE("ExecutePacketType0 overflow (read count {:08X}, packet count {:08X})",
         COMMAND_PROCESSOR::GetCurrentRingReadCount(),
         count * sizeof(uint32_t));
  return false;
}
/*
    Todo: optimize this function this one along with execute packet type III are
   the most frequently called functions for PM4
*/
XE_NOINLINE
bool COMMAND_PROCESSOR::ExecutePacketType0(uint32_t packet) XE_RESTRICT {
  // Type-0 packet.
  // Write count registers in sequence to the registers starting at
  // (base_index << 2).

  uint32_t count = ((packet >> 16) & 0x3FFF) + 1;

  if (COMMAND_PROCESSOR::GetCurrentRingReadCount() >=
      count * sizeof(uint32_t)) {
    trace_writer_.WritePacketStart(uint32_t(reader_.read_ptr() - 4), 1 + count);

    uint32_t base_index = (packet & 0x7FFF);
    uint32_t write_one_reg = (packet >> 15) & 0x1;

    if (!write_one_reg) {
      COMMAND_PROCESSOR::WriteRegisterRangeFromRing(&reader_, base_index,
                                                    count);

    } else {
      COMMAND_PROCESSOR::WriteOneRegisterFromRing(base_index, count);
    }

    trace_writer_.WritePacketEnd();
    return true;
  } else {
    return COMMAND_PROCESSOR::ExecutePacketType0_CountOverflow(count);
  }
}
XE_NOINLINE
bool COMMAND_PROCESSOR::ExecutePacketType1(uint32_t packet) XE_RESTRICT {
  // Type-1 packet.
  // Contains two registers of data. Type-0 should be more common.
  trace_writer_.WritePacketStart(uint32_t(reader_.read_ptr() - 4), 3);
  uint32_t reg_index_1 = packet & 0x7FF;
  uint32_t reg_index_2 = (packet >> 11) & 0x7FF;
  uint32_t reg_data_1 = reader_.ReadAndSwap<uint32_t>();
  uint32_t reg_data_2 = reader_.ReadAndSwap<uint32_t>();
  COMMAND_PROCESSOR::WriteRegister(reg_index_1, reg_data_1);
  COMMAND_PROCESSOR::WriteRegister(reg_index_2, reg_data_2);
  trace_writer_.WritePacketEnd();
  return true;
}

bool COMMAND_PROCESSOR::ExecutePacketType2(uint32_t packet) XE_RESTRICT {
  // Type-2 packet.
  // No-op. Do nothing.
  trace_writer_.WritePacketStart(uint32_t(reader_.read_ptr() - 4), 1);
  trace_writer_.WritePacketEnd();
  return true;
}
XE_FORCEINLINE
XE_NOALIAS
uint32_t COMMAND_PROCESSOR::GetCurrentRingReadCount() {
  return reader_.read_count();
}
XE_NOINLINE
XE_COLD
bool COMMAND_PROCESSOR::ExecutePacketType3_CountOverflow(uint32_t count) {
  XELOGE("ExecutePacketType3 overflow (read count {:08X}, packet count {:08X})",
         COMMAND_PROCESSOR::GetCurrentRingReadCount(),
         count * sizeof(uint32_t));
  return false;
}
XE_NOINLINE
bool COMMAND_PROCESSOR::ExecutePacketType3(uint32_t packet) XE_RESTRICT {
  // Type-3 packet.
  uint32_t opcode = (packet >> 8) & 0x7F;
  uint32_t count = ((packet >> 16) & 0x3FFF) + 1;
  auto data_start_offset = reader_.read_offset();

  if (COMMAND_PROCESSOR::GetCurrentRingReadCount() >=
      count * sizeof(uint32_t)) {
    // To handle nesting behavior when tracing we special case indirect buffers.
    if (opcode == PM4_INDIRECT_BUFFER) {
      trace_writer_.WritePacketStart(uint32_t(reader_.read_ptr() - 4), 2);
    } else {
      trace_writer_.WritePacketStart(uint32_t(reader_.read_ptr() - 4),
                                     1 + count);
    }

    // & 1 == predicate - when set, we do bin check to see if we should execute
    // the packet. Only type 3 packets are affected.
    // We also skip predicated swaps, as they are never valid (probably?).
    if (packet & 1) {
      bool any_pass = (bin_select_ & bin_mask_) != 0;
      if (!any_pass || opcode == PM4_XE_SWAP) {
        reader_.AdvanceRead(count * sizeof(uint32_t));
        trace_writer_.WritePacketEnd();
        return true;
      }
    }

    bool result = false;
    switch (opcode) {
      case PM4_ME_INIT:
        result = COMMAND_PROCESSOR::ExecutePacketType3_ME_INIT(packet, count);
        break;
      case PM4_NOP:
        result = COMMAND_PROCESSOR::ExecutePacketType3_NOP(packet, count);
        break;
      case PM4_INTERRUPT:
        result = COMMAND_PROCESSOR::ExecutePacketType3_INTERRUPT(packet, count);
        break;
      case PM4_XE_SWAP:
        result = COMMAND_PROCESSOR::ExecutePacketType3_XE_SWAP(packet, count);
        break;
      case PM4_INDIRECT_BUFFER:
      case PM4_INDIRECT_BUFFER_PFD:
        result = COMMAND_PROCESSOR::ExecutePacketType3_INDIRECT_BUFFER(packet,
                                                                       count);
        break;
      case PM4_WAIT_REG_MEM:
        result =
            COMMAND_PROCESSOR::ExecutePacketType3_WAIT_REG_MEM(packet, count);
        break;
      case PM4_REG_RMW:
        result = COMMAND_PROCESSOR::ExecutePacketType3_REG_RMW(packet, count);
        break;
      case PM4_REG_TO_MEM:
        result =
            COMMAND_PROCESSOR::ExecutePacketType3_REG_TO_MEM(packet, count);
        break;
      case PM4_MEM_WRITE:
        result = COMMAND_PROCESSOR::ExecutePacketType3_MEM_WRITE(packet, count);
        break;
      case PM4_COND_WRITE:
        result =
            COMMAND_PROCESSOR::ExecutePacketType3_COND_WRITE(packet, count);
        break;
      case PM4_EVENT_WRITE:
        result =
            COMMAND_PROCESSOR::ExecutePacketType3_EVENT_WRITE(packet, count);
        break;
      case PM4_EVENT_WRITE_SHD:
        result = COMMAND_PROCESSOR::ExecutePacketType3_EVENT_WRITE_SHD(packet,
                                                                       count);
        break;
      case PM4_EVENT_WRITE_EXT:
        result = COMMAND_PROCESSOR::ExecutePacketType3_EVENT_WRITE_EXT(packet,
                                                                       count);
        break;
      case PM4_EVENT_WRITE_ZPD:
        result = COMMAND_PROCESSOR::ExecutePacketType3_EVENT_WRITE_ZPD(packet,
                                                                       count);
        break;
      case PM4_DRAW_INDX:
        result = COMMAND_PROCESSOR::ExecutePacketType3_DRAW_INDX(packet, count);
        break;
      case PM4_DRAW_INDX_2:
        result =
            COMMAND_PROCESSOR::ExecutePacketType3_DRAW_INDX_2(packet, count);
        break;
      case PM4_SET_CONSTANT:
        result =
            COMMAND_PROCESSOR::ExecutePacketType3_SET_CONSTANT(packet, count);
        break;
      case PM4_SET_CONSTANT2:
        result =
            COMMAND_PROCESSOR::ExecutePacketType3_SET_CONSTANT2(packet, count);
        break;
      case PM4_LOAD_ALU_CONSTANT:
        result = COMMAND_PROCESSOR::ExecutePacketType3_LOAD_ALU_CONSTANT(packet,
                                                                         count);
        break;
      case PM4_SET_SHADER_CONSTANTS:
        result = COMMAND_PROCESSOR::ExecutePacketType3_SET_SHADER_CONSTANTS(
            packet, count);
        break;
      case PM4_IM_LOAD:
        result = COMMAND_PROCESSOR::ExecutePacketType3_IM_LOAD(packet, count);
        break;
      case PM4_IM_LOAD_IMMEDIATE:
        result = COMMAND_PROCESSOR::ExecutePacketType3_IM_LOAD_IMMEDIATE(packet,
                                                                         count);
        break;
      case PM4_INVALIDATE_STATE:
        result = COMMAND_PROCESSOR::ExecutePacketType3_INVALIDATE_STATE(packet,
                                                                        count);
        break;
      case PM4_VIZ_QUERY:
        result = COMMAND_PROCESSOR::ExecutePacketType3_VIZ_QUERY(packet, count);
        break;

      case PM4_SET_BIN_MASK_LO: {
        uint32_t value = reader_.ReadAndSwap<uint32_t>();
        bin_mask_ = (bin_mask_ & 0xFFFFFFFF00000000ull) | value;
        result = true;
      } break;
      case PM4_SET_BIN_MASK_HI: {
        uint32_t value = reader_.ReadAndSwap<uint32_t>();
        bin_mask_ =
            (bin_mask_ & 0xFFFFFFFFull) | (static_cast<uint64_t>(value) << 32);
        result = true;
      } break;
      case PM4_SET_BIN_SELECT_LO: {
        uint32_t value = reader_.ReadAndSwap<uint32_t>();
        bin_select_ = (bin_select_ & 0xFFFFFFFF00000000ull) | value;
        result = true;
      } break;
      case PM4_SET_BIN_SELECT_HI: {
        uint32_t value = reader_.ReadAndSwap<uint32_t>();
        bin_select_ = (bin_select_ & 0xFFFFFFFFull) |
                      (static_cast<uint64_t>(value) << 32);
        result = true;
      } break;
      case PM4_SET_BIN_MASK: {
        assert_true(count == 2);
        uint64_t val_hi = reader_.ReadAndSwap<uint32_t>();
        uint64_t val_lo = reader_.ReadAndSwap<uint32_t>();
        bin_mask_ = (val_hi << 32) | val_lo;
        result = true;
      } break;
      case PM4_SET_BIN_SELECT: {
        assert_true(count == 2);
        uint64_t val_hi = reader_.ReadAndSwap<uint32_t>();
        uint64_t val_lo = reader_.ReadAndSwap<uint32_t>();
        bin_select_ = (val_hi << 32) | val_lo;
        result = true;
      } break;
      case PM4_CONTEXT_UPDATE: {
        assert_true(count == 1);
        uint32_t value = reader_.ReadAndSwap<uint32_t>();
        XELOGGPU("GPU context update = {:08X}", value);
        assert_true(value == 0);
        result = true;
        break;
      }
      case PM4_WAIT_IB_PFD_COMPLETE: {
        // Waits for earlier INDIRECT_BUFFER_PFD base and size writes.
        // No wait is needed since we process indirect buffers synchronously.
        reader_.AdvanceRead(count * sizeof(uint32_t));
        result = true;
        break;
      }
      case PM4_WAIT_FOR_IDLE: {
        // This opcode is used by 5454084E while going / being ingame.
        assert_true(count == 1);
        uint32_t value = reader_.ReadAndSwap<uint32_t>();
        XELOGGPU("GPU wait for idle = {:08X}", value);
        result = true;
        break;
      }

      default:
        return COMMAND_PROCESSOR::HitUnimplementedOpcode(opcode, count);
    }

    trace_writer_.WritePacketEnd();
#if XE_ENABLE_TRACE_WRITER_INSTRUMENTATION == 1

    if (opcode == PM4_XE_SWAP) {
      // XE_GPU_SWAP_COUNTER_LOG=1: bounded XE_SWAP counter progression log so
      // unattended captures (trace_gpu_frame_at_swap consults THIS counter,
      // which runs on a different clock than the present/swap stats) can be
      // calibrated to wall time and scene from a single measurement run.
      static const bool s_swap_counter_log =
          std::getenv("XE_GPU_SWAP_COUNTER_LOG") != nullptr;
      if (s_swap_counter_log && (counter_ & 63u) == 0u) {
        XELOGI("Pm4SwapCounter counter={}", counter_);
      }
      // Unattended single-frame capture: arm the same state the Trace Frame
      // hotkey sets when the swap counter REACHES the requested value (0 =
      // off). >= not ==: the counter advances in jumps for titles that issue
      // multiple XE_SWAP-counted packets per presented frame (Sonic
      // Generations measured ~6x the present rate with visible skips), so an
      // exact match can never fire. The one-shot latch keeps this to a
      // single capture.
      static bool s_at_swap_fired = false;
      if (cvars::trace_gpu_frame_at_swap > 0 && !s_at_swap_fired &&
          counter_ >= uint32_t(cvars::trace_gpu_frame_at_swap) &&
          trace_state_ == TraceState::kDisabled && !trace_writer_.is_open()) {
        s_at_swap_fired = true;
        trace_state_ = TraceState::kSingleFrame;
        trace_frame_path_ = cvars::trace_gpu_prefix;
      }
      // End the trace writer frame.
      if (trace_writer_.is_open()) {
        trace_writer_.WriteEvent(EventCommand::Type::kSwap);
        trace_writer_.Flush();
        if (trace_state_ == TraceState::kSingleFrame) {
          trace_state_ = TraceState::kDisabled;
          CloseTrace();
        }
      } else if (trace_state_ == TraceState::kSingleFrame) {
        // New trace request - we only start tracing at the beginning of a
        // frame.
        auto executable_module = kernel_state_->GetExecutableModule();
        uint32_t title_id =
            executable_module ? executable_module->title_id() : 0;
        uint32_t frame_index = counter_ - 1;
        auto file_name = BuildTraceFileName(title_id, frame_index, false);
        auto path = trace_frame_path_ / file_name;
        if (OpenTrace(path, title_id, frame_index, false)) {
          InitializeTrace();
        }
      }
    }
#endif

    assert_true(reader_.read_offset() ==
                (data_start_offset + (count * sizeof(uint32_t))) %
                    reader_.capacity());
    return result;
  } else {
    return COMMAND_PROCESSOR::ExecutePacketType3_CountOverflow(count);
  }
}

XE_NOINLINE
XE_COLD
bool COMMAND_PROCESSOR::HitUnimplementedOpcode(uint32_t opcode,
                                               uint32_t count) XE_RESTRICT {
  XELOGGPU("Unimplemented GPU OPCODE: 0x{:02X}\t\tCOUNT: {}\n", opcode, count);
  assert_always();
  reader_.AdvanceRead(count * sizeof(uint32_t));
  trace_writer_.WritePacketEnd();
  return false;
}
XE_NOINLINE
bool COMMAND_PROCESSOR::ExecutePacketType3_ME_INIT(uint32_t packet,
                                                   uint32_t count) XE_RESTRICT {
  // initialize CP's micro-engine
  me_bin_.resize(count);
  for (uint32_t i = 0; i < count; i++) {
    me_bin_[i] = reader_.ReadAndSwap<uint32_t>();
  }
  return true;
}

bool COMMAND_PROCESSOR::ExecutePacketType3_NOP(uint32_t packet,
                                               uint32_t count) XE_RESTRICT {
  // skip N 32-bit words to get to the next packet
  // No-op, ignore some data.
  reader_.AdvanceRead(count * sizeof(uint32_t));
  return true;
}
XE_NOINLINE
bool COMMAND_PROCESSOR::ExecutePacketType3_INTERRUPT(
    uint32_t packet, uint32_t count) XE_RESTRICT {
  SCOPE_profile_cpu_f("gpu");

  // generate interrupt from the command stream
  uint32_t cpu_mask = reader_.ReadAndSwap<uint32_t>();
  if ((cpu_mask & 0x3Fu) &&
      !COMMAND_PROCESSOR::PrepareGuestMemoryForCompletionExposure()) {
    XELOGE("INTERRUPT: failed to publish deferred guest-memory producers");
    return false;
  }
  for (int n = 0; n < 6; n++) {
    if (cpu_mask & (1 << n)) {
      graphics_system_->DispatchInterruptCallback(1, n);
    }
  }
  return true;
}
XE_NOINLINE
bool COMMAND_PROCESSOR::ExecutePacketType3_XE_SWAP(uint32_t packet,
                                                   uint32_t count) XE_RESTRICT {
  SCOPE_profile_cpu_f("gpu");

  Profiler::Flip();

  // Xenia-specific VdSwap hook.
  // VdSwap will post this to tell us we need to swap the screen/fire an
  // interrupt.
  // 63 words here, but only the first has any data.
  uint32_t magic = reader_.ReadAndSwap<fourcc_t>();
  assert_true(magic == kSwapSignature);

  // TODO(benvanik): only swap frontbuffer ptr.
  uint32_t frontbuffer_ptr = reader_.ReadAndSwap<uint32_t>();
  uint32_t frontbuffer_width = reader_.ReadAndSwap<uint32_t>();
  uint32_t frontbuffer_height = reader_.ReadAndSwap<uint32_t>();
  reader_.AdvanceRead((count - 4) * sizeof(uint32_t));

  last_swap_width_ = frontbuffer_width;
  last_swap_height_ = frontbuffer_height;
  COMMAND_PROCESSOR::IssueSwap(frontbuffer_ptr, frontbuffer_width,
                               frontbuffer_height);
  LogFpsTick();

  ++counter_;
  return true;
}

bool COMMAND_PROCESSOR::ExecutePacketType3_INDIRECT_BUFFER(
    uint32_t packet, uint32_t count) XE_RESTRICT {
  // indirect buffer dispatch
  uint32_t list_ptr = CpuToGpu(reader_.ReadAndSwap<uint32_t>());
  uint32_t list_length = reader_.ReadAndSwap<uint32_t>();
  assert_zero(list_length & ~0xFFFFF);
  list_length &= 0xFFFFF;
  uint32_t list_physical = GpuToCpu(list_ptr);
  if (!COMMAND_PROCESSOR::PrepareGuestMemoryForHostRead(
          list_physical, list_length * sizeof(uint32_t))) {
    XELOGE("INDIRECT_BUFFER: failed to prepare guest range {:08X}(+{})",
           list_physical, list_length * sizeof(uint32_t));
    return false;
  }
  COMMAND_PROCESSOR::ExecuteIndirectBuffer(list_physical, list_length);
  return true;
}

/*
        chrispy: this is fine to inline, as a noinline function it compiled down
   to 54 bytes
*/
static bool MatchValueAndRef(uint32_t value, uint32_t ref, uint32_t wait_info) {
  // smaller code is generated than the #else path, although whether it is
  // faster i do not know. i don't think games do an enormous number of
  // cond_write though, so we have picked the path with the smaller codegen. we
  // do technically have more instructions executed vs the switch case method,
  // but we have no mispredicts and most of our instructions are 0.25/0.3
  // throughput
  return ((((value < ref) << 1) | ((value <= ref) << 2) |
           ((value == ref) << 3) | ((value != ref) << 4) |
           ((value >= ref) << 5) | ((value > ref) << 6) | (1 << 7)) >>
          (wait_info & 7)) &
         1;
}
XE_NOINLINE
bool COMMAND_PROCESSOR::ExecutePacketType3_WAIT_REG_MEM(
    uint32_t packet, uint32_t count) XE_RESTRICT {
  SCOPE_profile_cpu_f("gpu");

  // wait until a register or memory location is a specific value
  uint32_t wait_info = reader_.ReadAndSwap<uint32_t>();
  uint32_t poll_reg_addr = reader_.ReadAndSwap<uint32_t>();
  uint32_t ref = reader_.ReadAndSwap<uint32_t>();
  uint32_t mask = reader_.ReadAndSwap<uint32_t>();
  uint32_t wait = reader_.ReadAndSwap<uint32_t>();

  bool is_memory = (wait_info & 0x10) != 0;
  assert_true(is_memory || poll_reg_addr < RegisterFile::kRegisterCount);
  // Strip the packet's low endian bits, then resolve the single 512 MiB
  // physical alias (matching Memory::TranslatePhysical) used for the authority
  // boundary, the trace read, and the actual RAM read.
  uint32_t poll_physical = xenos::CpuToGpu(poll_reg_addr & ~uint32_t(0x3));
  const volatile uint32_t& value_ref =
      is_memory ? *reinterpret_cast<uint32_t*>(
                      memory_->TranslatePhysical(poll_physical))
                : register_file_->values[poll_reg_addr];

  if (is_memory) {
    if (!COMMAND_PROCESSOR::FlushPendingWritebacksForWait(poll_physical)) {
      XELOGE("WAIT_REG_MEM: failed to prepare guest word {:08X}", poll_physical);
      return false;
    }
  }

  bool matched = false;
  uint32_t wait_iterations = 0;

  // XE_CP_WAIT_CENSUS=1 (diagnostic, default off): per-window counts of
  // WAIT_REG_MEM packets, the ones that had to sleep, the sleeps, the
  // requested vs. actual sleep time, and the busiest poll addresses -- how
  // much of the command processor's idle time is guest latency versus the
  // whole-millisecond sleep quantum.
  static const bool s_wait_census = std::getenv("XE_CP_WAIT_CENSUS") != nullptr;
  std::chrono::steady_clock::time_point census_start;
  uint64_t census_requested_us = 0;
  const bool pacing_census = FramePacingCensusEnabled();
  if (s_wait_census || pacing_census) {
    census_start = std::chrono::steady_clock::now();
  }

  do {
    uint32_t value = value_ref;
    if (is_memory) {
      trace_writer_.WriteMemoryRead(poll_physical, sizeof(uint32_t));
      value = xenos::GpuSwap(value,
                             static_cast<xenos::Endian>(poll_reg_addr & 0x3));
    } else {
      if (poll_reg_addr == XE_GPU_REG_COHER_STATUS_HOST) {
        MakeCoherent();
        value = value_ref;
      }
    }
    matched = MatchValueAndRef(value & mask, ref, wait_info);

    if (!matched) {
      // Freeze forensics: a wait that has spun for ~5 seconds is almost
      // certainly a lost signal (whoever should write the value never did) —
      // log WHAT is being waited on, once per stuck wait.
      if (++wait_iterations == 500) {
        XELOGW(
            "WAIT_REG_MEM stuck ~5s: {} {:08X} value={:08X} mask={:08X} "
            "ref={:08X} cond={:X} wait={}",
            is_memory ? "mem" : "reg", poll_reg_addr, value, mask, ref,
            wait_info & 0x7, wait);
      }
      // Wait.
      if (wait >= 0x100) {
        PrepareForWait();
        if (!cvars::vsync) {
          // User wants it fast and dangerous.
          // do nothing
        } else {
          xe::threading::Sleep(std::chrono::milliseconds(wait / 0x100));
          census_requested_us += uint64_t(wait / 0x100) * 1000;
          ReturnFromWait();
        }

        if (!worker_running_) {
          // Short-circuited exit.
          return false;
        }
      } else {
      }
    }
  } while (!matched);

  if (pacing_census && wait_iterations) {
    GetFramePacingClock().cp_waited_ns.fetch_add(
        uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(
                     std::chrono::steady_clock::now() - census_start)
                     .count()),
        std::memory_order_relaxed);
  }
  if (s_wait_census) {
    struct AddrStat {
      uint32_t addr = 0, ref = 0, mask = 0, cond = 0;
      uint64_t packets = 0, missed = 0, sleeps = 0, wall_us = 0;
    };
    static std::vector<AddrStat> s_addrs;
    static uint64_t s_packets = 0, s_missed = 0, s_sleeps = 0;
    static uint64_t s_wall_us = 0, s_requested_us = 0;
    // Missed-wait wall time histogram, 2 ms buckets (the last is >= 16 ms):
    // a flat 0-16 ms spread means the wait absorbs vblank alignment, a narrow
    // peak means a fixed guest-side latency.
    static uint32_t s_wait_hist[9] = {};
    static auto s_window_start = std::chrono::steady_clock::now();
    auto now = std::chrono::steady_clock::now();
    uint64_t wall_us =
        uint64_t(std::chrono::duration_cast<std::chrono::microseconds>(
                     now - census_start)
                     .count());
    ++s_packets;
    if (wait_iterations) {
      ++s_missed;
      s_sleeps += wait_iterations;
      s_wall_us += wall_us;
      s_requested_us += census_requested_us;
      ++s_wait_hist[std::min<uint64_t>(wall_us / 2000, 8)];
    }
    AddrStat* stat = nullptr;
    for (AddrStat& a : s_addrs) {
      if (a.addr == poll_reg_addr && a.ref == ref && a.mask == mask) {
        stat = &a;
        break;
      }
    }
    if (!stat && s_addrs.size() < 64) {
      s_addrs.emplace_back();
      stat = &s_addrs.back();
      stat->addr = poll_reg_addr;
      stat->ref = ref;
      stat->mask = mask;
      stat->cond = (wait_info & 0x7) | (is_memory ? 0x10 : 0);
    }
    if (stat) {
      ++stat->packets;
      if (wait_iterations) {
        ++stat->missed;
        stat->sleeps += wait_iterations;
        stat->wall_us += wall_us;
      }
    }
    uint64_t window_us =
        uint64_t(std::chrono::duration_cast<std::chrono::microseconds>(
                     now - s_window_start)
                     .count());
    if (window_us >= 5000000) {
      std::sort(s_addrs.begin(), s_addrs.end(),
                [](const AddrStat& a, const AddrStat& b) {
                  return a.wall_us > b.wall_us;
                });
      std::string top;
      for (size_t i = 0; i < s_addrs.size() && i < 4; ++i) {
        const AddrStat& a = s_addrs[i];
        top += fmt::format(
            " [{}{:08X} ref={:08X} mask={:08X} cond={:X} packets={} "
            "missed={} sleeps={} wall_ms={:.1f}]",
            (a.cond & 0x10) ? "mem " : "reg ", a.addr, a.ref, a.mask,
            a.cond & 0x7, a.packets, a.missed, a.sleeps, a.wall_us / 1000.0);
      }
      XELOGI(
          "CpWaitCensus window_ms={:.0f}: packets={} missed={} sleeps={} "
          "waited_ms={:.1f} requested_ms={:.1f} ({:.1f}% of window){}",
          window_us / 1000.0, s_packets, s_missed, s_sleeps, s_wall_us / 1000.0,
          s_requested_us / 1000.0,
          100.0 * double(s_wall_us) / double(window_us), top);
      XELOGI(
          "CpWaitCensus missed-wait ms histogram (2 ms buckets): {} {} {} {} "
          "{} {} {} {} {}",
          s_wait_hist[0], s_wait_hist[1], s_wait_hist[2], s_wait_hist[3],
          s_wait_hist[4], s_wait_hist[5], s_wait_hist[6], s_wait_hist[7],
          s_wait_hist[8]);
      std::fill(std::begin(s_wait_hist), std::end(s_wait_hist), 0u);
      s_addrs.clear();
      s_packets = s_missed = s_sleeps = s_wall_us = s_requested_us = 0;
      s_window_start = now;
    }
  }

  return true;
}
XE_NOINLINE
bool COMMAND_PROCESSOR::ExecutePacketType3_REG_RMW(uint32_t packet,
                                                   uint32_t count) XE_RESTRICT {
  // register read/modify/write
  // ? (used during shader upload and edram setup)
  uint32_t rmw_info = reader_.ReadAndSwap<uint32_t>();
  uint32_t and_mask = reader_.ReadAndSwap<uint32_t>();
  uint32_t or_mask = reader_.ReadAndSwap<uint32_t>();
  uint32_t value = register_file_->values[rmw_info & 0x1FFF];
  if ((rmw_info >> 31) & 0x1) {
    // & reg
    value &= register_file_->values[and_mask & 0x1FFF];
  } else {
    // & imm
    value &= and_mask;
  }
  if ((rmw_info >> 30) & 0x1) {
    // | reg
    value |= register_file_->values[or_mask & 0x1FFF];
  } else {
    // | imm
    value |= or_mask;
  }
  COMMAND_PROCESSOR::WriteRegister(rmw_info & 0x1FFF, value);
  return true;
}

bool COMMAND_PROCESSOR::ExecutePacketType3_REG_TO_MEM(
    uint32_t packet, uint32_t count) XE_RESTRICT {
  // Copy Register to Memory (?)
  // Count is 2, assuming a Register Addr and a Memory Addr.

  uint32_t reg_addr = reader_.ReadAndSwap<uint32_t>();
  uint32_t mem_addr = reader_.ReadAndSwap<uint32_t>();

  uint32_t reg_val;

  assert_true(reg_addr < RegisterFile::kRegisterCount);
  reg_val = register_file_->values[reg_addr];

  auto endianness = static_cast<xenos::Endian>(mem_addr & 0x3);
  mem_addr &= ~0x3;
  reg_val = GpuSwap(reg_val, endianness);
  xe::store(memory_->TranslatePhysical(mem_addr), reg_val);
  trace_writer_.WriteMemoryWrite(CpuToGpu(mem_addr), 4);

  return true;
}
XE_NOINLINE
bool COMMAND_PROCESSOR::ExecutePacketType3_MEM_WRITE(
    uint32_t packet, uint32_t count) XE_RESTRICT {
  uint32_t write_addr = reader_.ReadAndSwap<uint32_t>();
  COMMAND_PROCESSOR::PrepareGuestMemoryForFenceWrite(
      GuestSignalKind::kMemWrite);
  for (uint32_t i = 0; i < count - 1; i++) {
    uint32_t write_data = reader_.ReadAndSwap<uint32_t>();

    auto endianness = static_cast<xenos::Endian>(write_addr & 0x3);
    auto addr = write_addr & ~0x3;
    write_data = GpuSwap(write_data, endianness);
    xe::store(memory_->TranslatePhysical(addr), write_data);
    trace_writer_.WriteMemoryWrite(CpuToGpu(addr), 4);
    write_addr += 4;
  }

  return true;
}
XE_NOINLINE
bool COMMAND_PROCESSOR::ExecutePacketType3_COND_WRITE(
    uint32_t packet, uint32_t count) XE_RESTRICT {
  // conditional write to memory or register
  uint32_t wait_info = reader_.ReadAndSwap<uint32_t>();
  uint32_t poll_reg_addr = reader_.ReadAndSwap<uint32_t>();
  uint32_t ref = reader_.ReadAndSwap<uint32_t>();
  uint32_t mask = reader_.ReadAndSwap<uint32_t>();
  uint32_t write_reg_addr = reader_.ReadAndSwap<uint32_t>();
  uint32_t write_data = reader_.ReadAndSwap<uint32_t>();
  uint32_t value;
  if (wait_info & 0x10) {
    // Memory.
    auto endianness = static_cast<xenos::Endian>(poll_reg_addr & 0x3);
    poll_reg_addr &= ~0x3;
    // Strip the packet's low endian bits, then resolve the single 512 MiB
    // physical alias used for the authority boundary, the trace read, and the
    // actual RAM read.
    uint32_t poll_physical = xenos::CpuToGpu(poll_reg_addr);
    if (!COMMAND_PROCESSOR::PrepareGuestMemoryForHostRead(poll_physical,
                                                          sizeof(uint32_t))) {
      XELOGE("COND_WRITE: failed to prepare guest predicate {:08X}",
             poll_physical);
      return false;
    }
    trace_writer_.WriteMemoryRead(poll_physical, 4);
    value = xe::load<uint32_t>(memory_->TranslatePhysical(poll_physical));
    value = GpuSwap(value, endianness);
  } else {
    // Register.
    assert_true(poll_reg_addr < RegisterFile::kRegisterCount);
    value = register_file_->values[poll_reg_addr];
  }
  bool matched = MatchValueAndRef(value & mask, ref, wait_info);

  if (matched) {
    // Write.
    if (wait_info & 0x100) {
      // Memory.
      COMMAND_PROCESSOR::PrepareGuestMemoryForFenceWrite(
          GuestSignalKind::kCondWrite);
      auto endianness = static_cast<xenos::Endian>(write_reg_addr & 0x3);
      write_reg_addr &= ~0x3;
      write_data = GpuSwap(write_data, endianness);
      xe::store(memory_->TranslatePhysical(write_reg_addr), write_data);
      trace_writer_.WriteMemoryWrite(CpuToGpu(write_reg_addr), 4);
    } else {
      // Register.
      COMMAND_PROCESSOR::WriteRegister(write_reg_addr, write_data);
    }
  }
  return true;
}
XE_FORCEINLINE
void COMMAND_PROCESSOR::WriteEventInitiator(uint32_t value) XE_RESTRICT {
  register_file_->values[XE_GPU_REG_VGT_EVENT_INITIATOR] = value;
}
bool COMMAND_PROCESSOR::ExecutePacketType3_EVENT_WRITE(
    uint32_t packet, uint32_t count) XE_RESTRICT {
  // generate an event that creates a write to memory when completed
  uint32_t initiator = reader_.ReadAndSwap<uint32_t>();
  // Writeback initiator.

  COMMAND_PROCESSOR::WriteEventInitiator(initiator & 0x3f);
  if (count == 1) {
    // Just an event flag? Where does this write?
  } else {
    // Write to an address.
    assert_always();
    reader_.AdvanceRead((count - 1) * sizeof(uint32_t));
  }
  return true;
}
XE_NOINLINE
bool COMMAND_PROCESSOR::ExecutePacketType3_EVENT_WRITE_SHD(
    uint32_t packet, uint32_t count) XE_RESTRICT {
  // generate a VS|PS_done event
  uint32_t initiator = reader_.ReadAndSwap<uint32_t>();
  uint32_t address = reader_.ReadAndSwap<uint32_t>();
  uint32_t value = reader_.ReadAndSwap<uint32_t>();
  // Writeback initiator.
  COMMAND_PROCESSOR::WriteEventInitiator(initiator & 0x3F);
  uint32_t data_value;
  if ((initiator >> 31) & 0x1) {
    // Write counter (GPU vblank counter?).
    data_value = counter_;
  } else {
    // Write value.
    data_value = value;
  }
  if (!COMMAND_PROCESSOR::PrepareGuestMemoryForCompletionExposure()) {
    XELOGE(
        "EVENT_WRITE_SHD: failed to publish deferred guest-memory producers");
    return false;
  }
  auto endianness = static_cast<xenos::Endian>(address & 0x3);
  address &= ~0x3;
  data_value = GpuSwap(data_value, endianness);
  uint8_t* write_destination = memory_->TranslatePhysical(address);
  if (address > 0x1FFFFFFF) {
    uint32_t writeback_base =
        register_file_->values[XE_GPU_REG_WRITEBACK_START];
    uint32_t writeback_size = register_file_->values[XE_GPU_REG_WRITEBACK_SIZE];
    uint32_t writeback_offset = address - writeback_base;
    // check whether the guest has written writeback base. if they haven't, skip
    // the offset check
    if (writeback_base != 0 && writeback_offset < writeback_size) {
      write_destination =
          memory_->TranslateVirtual(0x7F000000 + writeback_offset);
    }
  }
  xe::store(write_destination, data_value);
  trace_writer_.WriteMemoryWrite(CpuToGpu(address), 4);
  return true;
}

bool COMMAND_PROCESSOR::ExecutePacketType3_EVENT_WRITE_EXT(
    uint32_t packet, uint32_t count) XE_RESTRICT {
  // generate a screen extent event
  uint32_t initiator = reader_.ReadAndSwap<uint32_t>();
  uint32_t address = reader_.ReadAndSwap<uint32_t>();
  // Writeback initiator.
  COMMAND_PROCESSOR::WriteEventInitiator(initiator & 0x3F);
  auto endianness = static_cast<xenos::Endian>(address & 0x3);
  address &= ~0x3;

  if (!COMMAND_PROCESSOR::PrepareGuestMemoryForCompletionExposure()) {
    XELOGE(
        "EVENT_WRITE_EXT: failed to publish deferred guest-memory producers");
    return false;
  }

  // Let us hope we can fake this.
  // This callback tells the driver the xy coordinates affected by a previous
  // drawcall.
  // https://www.google.com/patents/US20060055701
  uint16_t extents[] = {
      byte_swap<unsigned short>(0 >> 3),  // min x
      byte_swap<unsigned short>(xenos::kTexture2DCubeMaxWidthHeight >>
                                3),       // max x
      byte_swap<unsigned short>(0 >> 3),  // min y
      byte_swap<unsigned short>(xenos::kTexture2DCubeMaxWidthHeight >>
                                3),  // max y
      byte_swap<unsigned short>(0),  // min z
      byte_swap<unsigned short>(1),  // max z
  };
  assert_true(endianness == xenos::Endian::k8in16);

  uint16_t* destination = (uint16_t*)memory_->TranslatePhysical(address);

  for (unsigned i = 0; i < 6; ++i) {
    destination[i] = extents[i];
  }

  trace_writer_.WriteMemoryWrite(CpuToGpu(address), sizeof(extents));
  return true;
}

XE_NOINLINE
bool COMMAND_PROCESSOR::ExecutePacketType3_EVENT_WRITE_ZPD(
    uint32_t packet, uint32_t count) XE_RESTRICT {
  assert_true(count == 1);
  uint32_t initiator = reader_.ReadAndSwap<uint32_t>();
  // Writeback initiator.
  COMMAND_PROCESSOR::WriteEventInitiator(initiator & 0x3F);

  uint32_t report_address =
      register_file_->values[XE_GPU_REG_RB_SAMPLE_COUNT_ADDR];
  uint32_t report_record_base = XenosZPDReport::GetRecordBase(report_address);
  // report_record_base already carries the record-aligned guest address (low
  // record/type bits stripped by GetRecordBase); keep it for the logical ZPD
  // begin/end sequencing below. Resolve the single 512 MiB physical alias
  // (matching Memory::TranslatePhysical) for the authority boundary and the
  // actual RAM pointer so both agree on the same word.
  uint32_t report_physical =
      report_record_base ? xenos::CpuToGpu(report_record_base) : 0;
  bool is_begin_record = XenosZPDReport::IsBeginRecord(report_address);
  bool is_end_record = XenosZPDReport::IsEndRecord(report_address);

  xe_gpu_depth_sample_counts* report =
      report_record_base
          ? memory_->TranslatePhysical<xe_gpu_depth_sample_counts*>(
                report_physical)
          : nullptr;

  if (report_record_base &&
      !COMMAND_PROCESSOR::PrepareGuestMemoryForHostRead(
          report_physical, sizeof(xe_gpu_depth_sample_counts))) {
    XELOGE("EVENT_WRITE_ZPD: failed to prepare guest report {:08X}",
           report_physical);
    return false;
  }

  // True if the record has the pending D3D sentinel.
  // Useful as a hint, but not authoritative for report boundaries.
  // QueryBatch titles can have multiple pending sentinels in a row and don't
  // necessarily update in an order we currently observe.
  bool guest_marks_end = report && XenosZPDReport::HasPendingSentinel(report);
  bool logical_active = zpd_active_segment_.logical_active;

  // QueryBatch fake fallback, which ignores record layout and just returns an
  // incrementing sample count on each event.
  if (cvars::occlusion_query_querybatch_range > 0) {
    uint32_t sample_count =
        XenosZPDReport::QueryBatchFakeSamples(querybatch_zpd_sample_count_);
    if (report) {
      // Both QueryBatch and conventional fake samples skip elective saturation.
      COMMAND_PROCESSOR::NotifyGuestRamWrite(
          report_physical, sizeof(xe_gpu_depth_sample_counts));
      XenosZPDReport::WriteSampleCount(report, sample_count, false);
    }
    return true;
  }

  if (GetZPDMode() != ZPDMode::kFake && !zpd_force_fake_fallback_) {
    if (logical_active && is_end_record) {
      COMMAND_PROCESSOR::EndZPDReport(report_address, false);
      return true;
    }
    if (is_begin_record) {
      // Clear the record so the game knows the BEGIN was processed and
      // stale sentinel data from a prior query lifetime doesn't persist.
      if (report) {
        COMMAND_PROCESSOR::NotifyGuestRamWrite(
            report_physical, sizeof(xe_gpu_depth_sample_counts));
        std::memset(report, 0, sizeof(xe_gpu_depth_sample_counts));
        // Guest-RAM write census (fix-replay-guest-ram-rollback 1.1): the
        // BEGIN clear writes the same record range the later report commit
        // writes. Diagnostic-only, default off.
        static const bool s_ram_write_census =
            std::getenv("XE_GPU_RAM_WRITE_CENSUS") != nullptr;
        if (s_ram_write_census) {
          XELOGI("RamWriteCensus writer=zpd_begin_clear base={:08X} len={}",
                 report_physical,
                 uint32_t(sizeof(xe_gpu_depth_sample_counts)));
        }
      }
      COMMAND_PROCESSOR::BeginZPDReport(report_address);
      return true;
    }
    if (!logical_active && is_end_record) {
      // No logical report is active for this slot, so this is likely an
      // orphaned END. In fast mode, replay the last cached delta so polling
      // code does not sit on the sentinel forever.
      if (GetZPDMode() == ZPDMode::kFast || GetZPDMode() == ZPDMode::kFastAlt) {
        uint32_t cached_delta = 1;
        auto cache_it = fast_zpd_report_cached_values_.find(report_record_base);
        if (cache_it != fast_zpd_report_cached_values_.end()) {
          cached_delta = cache_it->second;
        }
        COMMAND_PROCESSOR::WriteZPDReport(0, report_record_base, 0,
                                          cached_delta, false);
      } else {
        // In strict mode, just pump in case a previous report has resolved.
        COMMAND_PROCESSOR::PumpQueryResolves();
      }
      return true;
    }
    // Address is neither BEGIN nor END (non-standard layout). Fall through
    // to the fake path so the guest at least gets a result written rather
    // than leaving the sentinel in place forever.
  }

  // Conventional fake fallback, which only touches records marked as pending.
  if (cvars::occlusion_query_fake_lower_threshold < 0 || !report_record_base ||
      !guest_marks_end) {
    return true;
  }

  fake_zpd_sample_count_ =
      (fake_zpd_sample_count_ <=
       static_cast<uint32_t>(cvars::occlusion_query_fake_lower_threshold))
          ? static_cast<uint32_t>(cvars::occlusion_query_fake_upper_threshold)
          : fake_zpd_sample_count_ - 1;

  COMMAND_PROCESSOR::NotifyGuestRamWrite(report_physical,
                                         sizeof(xe_gpu_depth_sample_counts));
  XenosZPDReport::WriteSampleCount(report, fake_zpd_sample_count_, false);
  return true;
}

bool COMMAND_PROCESSOR::ExecutePacketType3Draw(
    uint32_t packet, const char* opcode_name, uint32_t viz_query_condition,
    uint32_t count_remaining) XE_RESTRICT {
  // if viz_query_condition != 0, this is a conditional draw based on viz query.
  // This ID matches the one issued in PM4_VIZ_QUERY
  // uint32_t viz_id = viz_query_condition & 0x3F;
  // when true, render conditionally based on query result
  // uint32_t viz_use = viz_query_condition & 0x100;

  assert_not_zero(count_remaining);
  if (!count_remaining) {
    XELOGE("{}: Packet too small, can't read VGT_DRAW_INITIATOR", opcode_name);
    return false;
  }
  reg::VGT_DRAW_INITIATOR vgt_draw_initiator;
  vgt_draw_initiator.value = reader_.ReadAndSwap<uint32_t>();
  --count_remaining;

  register_file_->values[XE_GPU_REG_VGT_DRAW_INITIATOR] =
      vgt_draw_initiator.value;
  bool draw_succeeded = true;
  // TODO(Triang3l): Remove IndexBufferInfo and replace handling of all this
  // with PrimitiveProcessor when the old Vulkan renderer is removed.
  bool is_indexed = false;
  IndexBufferInfo index_buffer_info;
  switch (vgt_draw_initiator.source_select) {
    case xenos::SourceSelect::kDMA: {
      // Indexed draw.
      is_indexed = true;

      // Two separate bounds checks so if there's only one missing register
      // value out of two, one uint32_t will be skipped in the command buffer,
      // not two.
      assert_not_zero(count_remaining);
      if (!count_remaining) {
        XELOGE("{}: Packet too small, can't read VGT_DMA_BASE", opcode_name);
        return false;
      }
      uint32_t vgt_dma_base = reader_.ReadAndSwap<uint32_t>();
      --count_remaining;
      register_file_->values[XE_GPU_REG_VGT_DMA_BASE] = vgt_dma_base;
      reg::VGT_DMA_SIZE vgt_dma_size;
      assert_not_zero(count_remaining);
      if (!count_remaining) {
        XELOGE("{}: Packet too small, can't read VGT_DMA_SIZE", opcode_name);
        return false;
      }
      vgt_dma_size.value = reader_.ReadAndSwap<uint32_t>();
      --count_remaining;
      register_file_->values[XE_GPU_REG_VGT_DMA_SIZE] = vgt_dma_size.value;

      uint32_t index_size_bytes =
          vgt_draw_initiator.index_size == xenos::IndexFormat::kInt16
              ? sizeof(uint16_t)
              : sizeof(uint32_t);
      // The base address must already be word-aligned according to the R6xx
      // documentation, but for safety.
      index_buffer_info.guest_base = vgt_dma_base & ~(index_size_bytes - 1);
      index_buffer_info.endianness = vgt_dma_size.swap_mode;
      index_buffer_info.format = vgt_draw_initiator.index_size;
      index_buffer_info.length = vgt_dma_size.num_words * index_size_bytes;
      index_buffer_info.count = vgt_draw_initiator.num_indices;
    } break;
    case xenos::SourceSelect::kImmediate: {
      // TODO(Triang3l): VGT_IMMED_DATA.
      XELOGE(
          "{}: Using immediate vertex indices, which are not supported yet. "
          "Report the game to Xenia developers!",
          opcode_name, uint32_t(vgt_draw_initiator.source_select));
      draw_succeeded = false;
      assert_always();
    } break;
    case xenos::SourceSelect::kAutoIndex: {
      // Auto draw.
      index_buffer_info.guest_base = 0;
      index_buffer_info.length = 0;
    } break;
    default: {
      // Invalid source selection.
      draw_succeeded = false;
      assert_unhandled_case(vgt_draw_initiator.source_select);
    } break;
  }

  // Skip to the next command, for example, if there are immediate indexes that
  // we don't support yet.
  reader_.AdvanceRead(count_remaining * sizeof(uint32_t));

  {
    // Upstream-delta census (XE_GPU_DELTA_CENSUS, 2026-10-09, Canary
    // 692cd59cf): viz_survey = coverage-only survey draws (dropped below),
    // viz_conditional = draws whose token makes them conditional on a VIZ ID
    // (currently always drawn).
    static const bool s_viz_draw_census =
        std::getenv("XE_GPU_DELTA_CENSUS") != nullptr;
    static uint32_t s_viz_survey_lines = 0, s_viz_conditional_lines = 0;
    static uint64_t s_viz_surveys = 0, s_viz_conditionals = 0;
    if (s_viz_draw_census && draw_succeeded) {
      auto census_viz = register_file_->Get<reg::PA_SC_VIZ_QUERY>();
      if (census_viz.viz_query_ena && census_viz.kill_pix_post_hi_z) {
        ++s_viz_surveys;
        if (s_viz_survey_lines < 4000 &&
            (s_viz_surveys & (s_viz_surveys - 1)) == 0) {
          ++s_viz_survey_lines;
          XELOGI("DeltaCensus viz_survey id={} total={}",
                 uint32_t(census_viz.viz_query_id), s_viz_surveys);
        }
      }
      if (viz_query_condition & 0x100) {
        ++s_viz_conditionals;
        if (s_viz_conditional_lines < 4000 &&
            (s_viz_conditionals & (s_viz_conditionals - 1)) == 0) {
          ++s_viz_conditional_lines;
          XELOGI("DeltaCensus viz_conditional id={} total={}",
                 viz_query_condition & 0x3F, s_viz_conditionals);
        }
      }
    }
  }
  if (draw_succeeded) {
    auto viz_query = register_file_->Get<reg::PA_SC_VIZ_QUERY>();
    if (!(viz_query.viz_query_ena && viz_query.kill_pix_post_hi_z)) {
      // Draw-range gate for trace-replay bisection (no effect at defaults).
      uint32_t draw_index = trace_draw_index_++;
      // Sub-draw index window for one draw (diagnostic: isolate strip
      // segments/triangles without guest memory injection, which has
      // page-granular cache-invalidation side effects on neighbors).
      bool window_skip_draw = false;
      if (is_indexed && !cvars::replay_index_window.empty()) {
        uint32_t window_draw = 0, window_start = 0, window_count = 0;
        if (std::sscanf(cvars::replay_index_window.c_str(), "%u,%u,%u",
                        &window_draw, &window_start, &window_count) == 3 &&
            draw_index == window_draw) {
          uint32_t index_size_bytes =
              index_buffer_info.format == xenos::IndexFormat::kInt16
                  ? sizeof(uint16_t)
                  : sizeof(uint32_t);
          index_buffer_info.guest_base += window_start * index_size_bytes;
          index_buffer_info.length -= window_start * index_size_bytes;
          index_buffer_info.count = window_count;
          vgt_draw_initiator.num_indices = window_count;
          window_skip_draw = window_count == 0;
        }
      }
      if (!cvars::replay_inject_memory.empty()) {
        // hexaddr,afterdraw,filepath[;...] - substitute guest memory before
        // the given draw (cross-GPU divergence attribution diagnostics).
        static uint32_t injections_done = 0;
        std::stringstream inject_specs(cvars::replay_inject_memory);
        std::string inject_spec;
        uint32_t inject_spec_index = 0;
        while (std::getline(inject_specs, inject_spec, ';')) {
          uint32_t inject_spec_bit = uint32_t(1) << inject_spec_index;
          ++inject_spec_index;
          uint32_t inject_addr = 0, inject_before_draw = 0;
          int inject_path_offset = 0;
          if ((injections_done & inject_spec_bit) ||
              std::sscanf(inject_spec.c_str(), "%x,%u,%n", &inject_addr,
                          &inject_before_draw, &inject_path_offset) < 2 ||
              draw_index != inject_before_draw) {
            continue;
          }
          injections_done |= inject_spec_bit;
          const char* inject_path = inject_spec.c_str() + inject_path_offset;
          FILE* inject_file = std::fopen(inject_path, "rb");
          if (!inject_file) {
            XELOGE("REPLAY: inject: failed to open {}", inject_path);
            continue;
          }
          std::fseek(inject_file, 0, SEEK_END);
          long inject_size = std::ftell(inject_file);
          std::fseek(inject_file, 0, SEEK_SET);
          if (inject_size > 0) {
            size_t inject_read = std::fread(
                memory_->TranslatePhysical(inject_addr), 1,
                size_t(inject_size), inject_file);
            TracePlaybackWroteMemory(inject_addr, uint32_t(inject_read));
            XELOGI("REPLAY: injected {} bytes at {:08X} before draw {}",
                   inject_read, inject_addr, draw_index);
          }
          std::fclose(inject_file);
        }
      }
      if (cvars::replay_draw_log) {
        auto vs = active_vertex_shader_;
        auto ps = active_pixel_shader_;
        auto rb_modecontrol = register_file_->Get<reg::RB_MODECONTROL>();
        XELOGI("REPLAYDRAW {} prim={} indices={} edram={} src={} vs={:016X} "
               "ps={:016X}",
               draw_index, uint32_t(vgt_draw_initiator.prim_type),
               vgt_draw_initiator.num_indices,
               uint32_t(rb_modecontrol.edram_mode),
               uint32_t(vgt_draw_initiator.source_select),
               vs ? vs->ucode_data_hash() : 0,
               ps ? ps->ucode_data_hash() : 0);
        {
          auto rb_depthcontrol = register_file_->Get<reg::RB_DEPTHCONTROL>();
          auto rb_depth_info = register_file_->Get<reg::RB_DEPTH_INFO>();
          auto rb_surface_info = register_file_->Get<reg::RB_SURFACE_INFO>();
          auto rb_color_info = register_file_->Get<reg::RB_COLOR_INFO>();
          auto pa_su_sc_mode_cntl =
              register_file_->Get<reg::PA_SU_SC_MODE_CNTL>();
          XELOGI("  DEPTH z_en={} z_wr={} zfunc={} depth_base={} "
                 "depth_fmt={} msaa={} pitch={} color0_base={} color0_fmt={}",
                 uint32_t(rb_depthcontrol.z_enable),
                 uint32_t(rb_depthcontrol.z_write_enable),
                 uint32_t(rb_depthcontrol.zfunc),
                 uint32_t(rb_depth_info.depth_base),
                 uint32_t(rb_depth_info.depth_format),
                 uint32_t(rb_surface_info.msaa_samples),
                 uint32_t(rb_surface_info.surface_pitch),
                 uint32_t(rb_color_info.color_base),
                 uint32_t(rb_color_info.color_format));
          XELOGI("  POLYOFS front_en={} back_en={} para_en={} "
                 "front_scale={} front_ofs={} back_scale={} back_ofs={} "
                 "vtx_window_ofs_en={}",
                 uint32_t(pa_su_sc_mode_cntl.poly_offset_front_enable),
                 uint32_t(pa_su_sc_mode_cntl.poly_offset_back_enable),
                 uint32_t(pa_su_sc_mode_cntl.poly_offset_para_enable),
                 register_file_->Get<float>(
                     XE_GPU_REG_PA_SU_POLY_OFFSET_FRONT_SCALE),
                 register_file_->Get<float>(
                     XE_GPU_REG_PA_SU_POLY_OFFSET_FRONT_OFFSET),
                 register_file_->Get<float>(
                     XE_GPU_REG_PA_SU_POLY_OFFSET_BACK_SCALE),
                 register_file_->Get<float>(
                     XE_GPU_REG_PA_SU_POLY_OFFSET_BACK_OFFSET),
                 uint32_t(pa_su_sc_mode_cntl.vtx_window_offset_enable));
          if (is_indexed) {
            XELOGI("  INDEXBUF base={:08X} len={} count={} fmt={} endian={}",
                   index_buffer_info.guest_base, index_buffer_info.length,
                   index_buffer_info.count,
                   uint32_t(index_buffer_info.format),
                   uint32_t(index_buffer_info.endianness));
          }
          for (uint32_t vf = 0; vf < 96; ++vf) {
            xenos::xe_gpu_vertex_fetch_t vfetch =
                register_file_->GetVertexFetch(vf);
            if (vfetch.type != xenos::FetchConstantType::kVertex ||
                !vfetch.size) {
              continue;
            }
            XELOGI("  VFETCH{} addr={:08X} size_dw={} endian={}", vf,
                   vfetch.address << 2, vfetch.size, uint32_t(vfetch.endian));
          }
        }
        if (rb_modecontrol.edram_mode == xenos::EdramMode::kCopy) {
          auto copy_control = register_file_->Get<reg::RB_COPY_CONTROL>();
          auto copy_dest_info = register_file_->Get<reg::RB_COPY_DEST_INFO>();
          auto copy_dest_pitch =
              register_file_->Get<reg::RB_COPY_DEST_PITCH>();
          XELOGI("  COPY dest={:08X} pitch={} height={} fmt={} src_sel={} "
                 "sample_sel={} clear_c={} clear_d={}",
                 register_file_->values[XE_GPU_REG_RB_COPY_DEST_BASE],
                 uint32_t(copy_dest_pitch.copy_dest_pitch),
                 uint32_t(copy_dest_pitch.copy_dest_height),
                 uint32_t(copy_dest_info.copy_dest_format),
                 uint32_t(copy_control.copy_src_select),
                 uint32_t(copy_control.copy_sample_select),
                 uint32_t(copy_control.color_clear_enable),
                 uint32_t(copy_control.depth_clear_enable));
        }
        for (uint32_t ci = 240; ci < 260; ++ci) {
          // Both potential ALU constant banks (VS base c0, PS base c256) -
          // which one the pixel shader actually addresses depends on the SQ
          // program config; dump both for offline ALU analysis.
          uint32_t reg_lo = XE_GPU_REG_SHADER_CONSTANT_000_X + ci * 4;
          uint32_t reg_hi = XE_GPU_REG_SHADER_CONSTANT_000_X + (256 + ci) * 4;
          XELOGI("  FCONST c{} lo=({},{},{},{}) hi=({},{},{},{})", ci,
                 register_file_->Get<float>(reg_lo),
                 register_file_->Get<float>(reg_lo + 1),
                 register_file_->Get<float>(reg_lo + 2),
                 register_file_->Get<float>(reg_lo + 3),
                 register_file_->Get<float>(reg_hi),
                 register_file_->Get<float>(reg_hi + 1),
                 register_file_->Get<float>(reg_hi + 2),
                 register_file_->Get<float>(reg_hi + 3));
        }
        for (uint32_t tf = 0; tf < 32; ++tf) {
          xenos::xe_gpu_texture_fetch_t fetch =
              register_file_->GetTextureFetch(tf);
          if (fetch.type != xenos::FetchConstantType::kTexture) {
            continue;
          }
          XELOGI("  FETCH{} dim={} minf={} magf={} mipf={} aniso={} "
                 "base={:08X} w={} h={} fmt={} tiled={} endian={} pitch={} "
                 "clampx={} clampy={}",
                 tf, uint32_t(fetch.dimension), uint32_t(fetch.min_filter),
                 uint32_t(fetch.mag_filter), uint32_t(fetch.mip_filter),
                 uint32_t(fetch.aniso_filter),
                 uint32_t(fetch.base_address) << 12,
                 fetch.size_2d.width + 1, fetch.size_2d.height + 1,
                 uint32_t(fetch.format), uint32_t(fetch.tiled),
                 uint32_t(fetch.endianness), fetch.pitch << 5,
                 uint32_t(fetch.clamp_x), uint32_t(fetch.clamp_y));
        }
      }
      if (!cvars::capture_dir.empty()) {
        // Structured per-draw manifest row (design D1): the same guest/
        // register state as replay_draw_log, emitted as one byte-stable JSON
        // line into <capture_dir>/manifest.jsonl. Floats are written as raw
        // register bits (hex), not decimal, so the row is byte-identical
        // across machines/arch (the D4 determinism gate depends on this).
        auto vs = active_vertex_shader_;
        auto ps = active_pixel_shader_;
        auto rb_modecontrol = register_file_->Get<reg::RB_MODECONTROL>();
        auto rb_depthcontrol = register_file_->Get<reg::RB_DEPTHCONTROL>();
        auto rb_depth_info = register_file_->Get<reg::RB_DEPTH_INFO>();
        auto rb_surface_info = register_file_->Get<reg::RB_SURFACE_INFO>();
        auto rb_color_info = register_file_->Get<reg::RB_COLOR_INFO>();
        auto pa_su_sc_mode_cntl = register_file_->Get<reg::PA_SU_SC_MODE_CNTL>();
        auto fbits = [&](uint32_t reg) -> uint32_t {
          return register_file_->values[reg];
        };
        std::string row = fmt::format(
            "{{\"draw\":{},\"prim\":{},\"indices\":{},\"edram\":{},\"src\":{}",
            draw_index, uint32_t(vgt_draw_initiator.prim_type),
            uint32_t(vgt_draw_initiator.num_indices),
            uint32_t(rb_modecontrol.edram_mode),
            uint32_t(vgt_draw_initiator.source_select));
        row += fmt::format(",\"vs\":\"{:016X}\",\"ps\":\"{:016X}\"",
                           vs ? vs->ucode_data_hash() : 0,
                           ps ? ps->ucode_data_hash() : 0);
        // Primitive-reset and cull state: a strip whose 16-bit indices carry
        // 0xFFFF renders differently depending on whether the guest enabled
        // reset (restart) or expects a degenerate out-of-range vertex; on hosts
        // that cannot disable restart the post-marker winding parity flips and
        // backface culling drops the flipped primitives.
        row += fmt::format(
            ",\"reset\":{{\"en\":{},\"indx\":\"{:08X}\"}}"
            ",\"cull\":{{\"front\":{},\"back\":{},\"face\":{}}}",
            uint32_t(pa_su_sc_mode_cntl.multi_prim_ib_ena),
            fbits(XE_GPU_REG_VGT_MULTI_PRIM_IB_RESET_INDX),
            uint32_t(pa_su_sc_mode_cntl.cull_front),
            uint32_t(pa_su_sc_mode_cntl.cull_back),
            uint32_t(pa_su_sc_mode_cntl.face));
        // Blend / alpha-test / scissor / viewport raw words, so a draw that
        // changes no destination pixel can be classified offline (blended to
        // zero, alpha-tested away, scissored out, or off-viewport) without
        // re-running the trace under a debugger.
        row += fmt::format(
            ",\"blend0\":\"{:08X}\",\"colorcontrol\":\"{:08X}\","
            "\"scissor\":[\"{:08X}\",\"{:08X}\"],\"vport\":[\"{:08X}\","
            "\"{:08X}\",\"{:08X}\",\"{:08X}\",\"{:08X}\",\"{:08X}\"],"
            "\"vte\":\"{:08X}\",\"clip\":\"{:08X}\"",
            fbits(XE_GPU_REG_RB_BLENDCONTROL0), fbits(XE_GPU_REG_RB_COLORCONTROL),
            fbits(XE_GPU_REG_PA_SC_WINDOW_SCISSOR_TL),
            fbits(XE_GPU_REG_PA_SC_WINDOW_SCISSOR_BR),
            fbits(XE_GPU_REG_PA_CL_VPORT_XSCALE), fbits(XE_GPU_REG_PA_CL_VPORT_XOFFSET),
            fbits(XE_GPU_REG_PA_CL_VPORT_YSCALE), fbits(XE_GPU_REG_PA_CL_VPORT_YOFFSET),
            fbits(XE_GPU_REG_PA_CL_VPORT_ZSCALE), fbits(XE_GPU_REG_PA_CL_VPORT_ZOFFSET),
            fbits(XE_GPU_REG_PA_CL_VTE_CNTL), fbits(XE_GPU_REG_PA_CL_CLIP_CNTL));
        row += fmt::format(
            ",\"depth\":{{\"z_en\":{},\"z_wr\":{},\"zfunc\":{},\"base\":{},"
            "\"fmt\":{}}}",
            uint32_t(rb_depthcontrol.z_enable),
            uint32_t(rb_depthcontrol.z_write_enable),
            uint32_t(rb_depthcontrol.zfunc),
            uint32_t(rb_depth_info.depth_base),
            uint32_t(rb_depth_info.depth_format));
        row += fmt::format(
            ",\"stencil\":{{\"en\":{},\"bf_en\":{},\"dc\":\"{:08X}\","
            "\"refmask\":\"{:08X}\",\"refmask_bf\":\"{:08X}\"}}"
            ",\"colormask\":\"{:08X}\"",
            uint32_t(rb_depthcontrol.stencil_enable),
            uint32_t(rb_depthcontrol.backface_enable), rb_depthcontrol.value,
            fbits(XE_GPU_REG_RB_STENCILREFMASK),
            fbits(XE_GPU_REG_RB_STENCILREFMASK_BF),
            fbits(XE_GPU_REG_RB_COLOR_MASK));
        row += fmt::format(
            ",\"surface\":{{\"msaa\":{},\"pitch\":{}}},"
            "\"color0\":{{\"base\":{},\"fmt\":{}}}",
            uint32_t(rb_surface_info.msaa_samples),
            uint32_t(rb_surface_info.surface_pitch),
            uint32_t(rb_color_info.color_base),
            uint32_t(rb_color_info.color_format));
        row += fmt::format(
            ",\"polyofs\":{{\"front_en\":{},\"back_en\":{},\"para_en\":{},"
            "\"front_scale\":\"{:08X}\",\"front_ofs\":\"{:08X}\","
            "\"back_scale\":\"{:08X}\",\"back_ofs\":\"{:08X}\","
            "\"vtx_win_ofs_en\":{}}}",
            uint32_t(pa_su_sc_mode_cntl.poly_offset_front_enable),
            uint32_t(pa_su_sc_mode_cntl.poly_offset_back_enable),
            uint32_t(pa_su_sc_mode_cntl.poly_offset_para_enable),
            fbits(XE_GPU_REG_PA_SU_POLY_OFFSET_FRONT_SCALE),
            fbits(XE_GPU_REG_PA_SU_POLY_OFFSET_FRONT_OFFSET),
            fbits(XE_GPU_REG_PA_SU_POLY_OFFSET_BACK_SCALE),
            fbits(XE_GPU_REG_PA_SU_POLY_OFFSET_BACK_OFFSET),
            uint32_t(pa_su_sc_mode_cntl.vtx_window_offset_enable));
        // XE_CAPTURE_CONSTS=<draw>: raw shader constant banks at that draw
        // (vertex float c0-c255, pixel float c256-c511, bool, loop) to
        // <capture_dir>/consts_<draw>.bin, so skinning / dynamic-index inputs
        // can be inspected offline (NaN / zero bone matrices etc.).
        static const char* s_consts_draw = std::getenv("XE_CAPTURE_CONSTS");
        if (s_consts_draw && draw_index == uint32_t(atoi(s_consts_draw))) {
          std::filesystem::path consts_path =
              std::filesystem::path(cvars::capture_dir) /
              fmt::format("consts_{:06d}.bin", draw_index);
          if (FILE* cf = xe::filesystem::OpenFile(consts_path, "wb")) {
            fwrite(&register_file_->values[XE_GPU_REG_SHADER_CONSTANT_000_X], 4,
                   512 * 4, cf);
            fwrite(&register_file_->values[XE_GPU_REG_SHADER_CONSTANT_BOOL_000_031],
                   4, 8, cf);
            fwrite(&register_file_->values[XE_GPU_REG_SHADER_CONSTANT_LOOP_00], 4,
                   32, cf);
            fclose(cf);
          }
        }
        if (is_indexed) {
          row += fmt::format(
              ",\"index\":{{\"base\":\"{:08X}\",\"len\":{},\"count\":{},"
              "\"fmt\":{},\"endian\":{}}}",
              index_buffer_info.guest_base, index_buffer_info.length,
              index_buffer_info.count, uint32_t(index_buffer_info.format),
              uint32_t(index_buffer_info.endianness));
        }
        row += ",\"fetch\":[";
        bool first_fetch = true;
        for (uint32_t tf = 0; tf < 32; ++tf) {
          xenos::xe_gpu_texture_fetch_t fetch =
              register_file_->GetTextureFetch(tf);
          if (fetch.type != xenos::FetchConstantType::kTexture) {
            continue;
          }
          row += fmt::format(
              "{}{{\"slot\":{},\"base\":\"{:08X}\",\"w\":{},\"h\":{},"
              "\"fmt\":{},\"tiled\":{},\"endian\":{},\"pitch\":{}}}",
              first_fetch ? "" : ",", tf,
              uint32_t(fetch.base_address) << 12, fetch.size_2d.width + 1,
              fetch.size_2d.height + 1, uint32_t(fetch.format),
              uint32_t(fetch.tiled), uint32_t(fetch.endianness),
              fetch.pitch << 5);
          first_fetch = false;
        }
        row += "]}";
        WriteCaptureManifestRow(row);
      }
      bool skip_range = int32_t(draw_index) >= cvars::replay_draw_skip_first &&
                        int32_t(draw_index) <= cvars::replay_draw_skip_last &&
                        cvars::replay_draw_skip_first >= 0;
      if (cvars::replay_skip_depth_only &&
          register_file_->Get<reg::RB_MODECONTROL>().edram_mode ==
              xenos::EdramMode::kDepthOnly) {
        skip_range = true;
      }
      if (window_skip_draw) {
        skip_range = true;
      }
      if (((int32_t(draw_index) < cvars::replay_draw_first ||
            int32_t(draw_index) > cvars::replay_draw_last) &&
           int32_t(draw_index) != cvars::replay_draw_include) ||
          skip_range) {
        // Skipped for bisection; packet already fully consumed above.
      } else if (int32_t(draw_index) == cvars::replay_zfunc_always_draw ||
                 int32_t(draw_index) == cvars::replay_solid_draw) {
        // Diagnostics (composable): zfunc-always bypasses the depth test to
        // distinguish depth-test failures from rasterization coverage gaps;
        // solid-draw forces RT0 blending to REPLACE and disables the
        // fixed-function alpha test / alpha-to-mask so any rasterized pixel
        // shows the raw fragment color (absent fragments - cull/clip/
        // coverage - stay unchanged).
        uint32_t depthcontrol_value =
            register_file_->values[XE_GPU_REG_RB_DEPTHCONTROL];
        uint32_t blendcontrol_value =
            register_file_->values[XE_GPU_REG_RB_BLENDCONTROL0];
        uint32_t colorcontrol_value =
            register_file_->values[XE_GPU_REG_RB_COLORCONTROL];
        if (int32_t(draw_index) == cvars::replay_zfunc_always_draw) {
          reg::RB_DEPTHCONTROL depthcontrol_always;
          depthcontrol_always.value = depthcontrol_value;
          depthcontrol_always.zfunc = xenos::CompareFunction::kAlways;
          register_file_->values[XE_GPU_REG_RB_DEPTHCONTROL] =
              depthcontrol_always.value;
        }
        if (int32_t(draw_index) == cvars::replay_solid_draw) {
          reg::RB_BLENDCONTROL blendcontrol_replace;
          blendcontrol_replace.value = 0;
          blendcontrol_replace.color_srcblend = xenos::BlendFactor::kOne;
          blendcontrol_replace.color_comb_fcn = xenos::BlendOp::kAdd;
          blendcontrol_replace.color_destblend = xenos::BlendFactor::kZero;
          blendcontrol_replace.alpha_srcblend = xenos::BlendFactor::kOne;
          blendcontrol_replace.alpha_comb_fcn = xenos::BlendOp::kAdd;
          blendcontrol_replace.alpha_destblend = xenos::BlendFactor::kZero;
          register_file_->values[XE_GPU_REG_RB_BLENDCONTROL0] =
              blendcontrol_replace.value;
          reg::RB_COLORCONTROL colorcontrol_no_kill;
          colorcontrol_no_kill.value = colorcontrol_value;
          colorcontrol_no_kill.alpha_test_enable = 0;
          colorcontrol_no_kill.alpha_to_mask_enable = 0;
          register_file_->values[XE_GPU_REG_RB_COLORCONTROL] =
              colorcontrol_no_kill.value;
        }
        draw_succeeded = COMMAND_PROCESSOR::IssueDraw(
            vgt_draw_initiator.prim_type, vgt_draw_initiator.num_indices,
            is_indexed ? &index_buffer_info : nullptr,
            xenos::IsMajorModeExplicit(vgt_draw_initiator.major_mode,
                                       vgt_draw_initiator.prim_type));
        register_file_->values[XE_GPU_REG_RB_DEPTHCONTROL] =
            depthcontrol_value;
        register_file_->values[XE_GPU_REG_RB_BLENDCONTROL0] =
            blendcontrol_value;
        register_file_->values[XE_GPU_REG_RB_COLORCONTROL] =
            colorcontrol_value;
      } else {
        // TODO(Triang3l): Don't drop the draw call completely if the vertex
        // shader has memexport.
        // TODO(Triang3l || JoelLinn): Handle this properly in the render
        // backends.
        // Backend-agnostic diagnostic (XE_GPU_FORCE_FLOAT=<reg>:<x>,<y>,<z>,<w>
        // with reg 0..511 = SHADER_CONSTANT_<reg>): pin one float constant
        // for every draw, restored right after, so constant/interpolant flow
        // can be A/B'd identically on every backend (dual-backend
        // attribution; the Metal-only XE_METAL_FORCE_FLOAT is its sibling).
        struct ForceFloatConstant {
          bool set = false;
          uint32_t reg = 0;
          uint32_t words[4] = {};
        };
        static const ForceFloatConstant s_force_float = [] {
          ForceFloatConstant f;
          const char* v = std::getenv("XE_GPU_FORCE_FLOAT");
          float x = 0, y = 0, z = 0, w = 0;
          unsigned reg = 0;
          if (v && std::sscanf(v, "%u:%f,%f,%f,%f", &reg, &x, &y, &z, &w) ==
                       5 &&
              reg < 512) {
            f.set = true;
            f.reg = XE_GPU_REG_SHADER_CONSTANT_000_X + reg * 4;
            float values[4] = {x, y, z, w};
            std::memcpy(f.words, values, sizeof(f.words));
          }
          return f;
        }();
        // Sibling (XE_GPU_FORCE_REG=<hex reg>:<hex value>[,...]): pin up to
        // four raw registers for every draw (e.g. 2200:0 = RB_DEPTHCONTROL
        // with depth and stencil tests off), restored right after.
        // Each entry may carry its own draw scope: <hex reg>:<hex value>@<draw>
        // (no suffix = every draw, or the XE_GPU_FORCE_REG_AT draw when set).
        struct ForceReg {
          uint32_t count = 0;
          uint32_t reg[8] = {};
          uint32_t value[8] = {};
          int64_t at[8] = {};
        };
        static const ForceReg s_force_reg = [] {
          ForceReg f;
          const char* v = std::getenv("XE_GPU_FORCE_REG");
          while (v && *v && f.count < 8) {
            unsigned reg = 0, value = 0;
            int consumed = 0;
            if (std::sscanf(v, "%x:%x%n", &reg, &value, &consumed) != 2 ||
                reg >= RegisterFile::kRegisterCount) {
              break;
            }
            f.reg[f.count] = reg;
            f.value[f.count] = value;
            f.at[f.count] = -1;
            v += consumed;
            if (*v == '@') {
              f.at[f.count] = std::atoll(v + 1);
              while (*v && *v != ',') {
                ++v;
              }
            }
            ++f.count;
            if (*v == ',') {
              ++v;
            }
          }
          return f;
        }();
        // XE_GPU_FORCE_REG_AT=<draw>: apply the register force at that draw
        // only (frame-wide forcing perturbs every earlier pass).
        static const int64_t s_force_reg_at = [] {
          const char* v = std::getenv("XE_GPU_FORCE_REG_AT");
          return v ? int64_t(std::atoll(v)) : int64_t(-1);
        }();
        bool force_reg_applied[8] = {};
        uint32_t force_reg_saved[8] = {};
        for (uint32_t i = 0; i < s_force_reg.count; ++i) {
          const int64_t at = s_force_reg.at[i] >= 0 ? s_force_reg.at[i]
                                                     : s_force_reg_at;
          if (at >= 0 && int64_t(draw_index) != at) {
            continue;
          }
          force_reg_applied[i] = true;
          force_reg_saved[i] = register_file_->values[s_force_reg.reg[i]];
          // Through the virtual write so backend dirty tracking (Vulkan's
          // constant-buffer up-to-date bits) sees the change.
          COMMAND_PROCESSOR::WriteRegister(s_force_reg.reg[i],
                                           s_force_reg.value[i]);
        }
        // Diagnostic (XE_GPU_DUMP_REGS_AT=<draw>): print the render-state
        // registers a draw is issued with (backend-independent values).
        static const int64_t s_dump_regs_at = [] {
          const char* v = std::getenv("XE_GPU_DUMP_REGS_AT");
          return v ? int64_t(std::atoll(v)) : int64_t(-1);
        }();
        if (int64_t(draw_index) == s_dump_regs_at) {
          const uint32_t* rv = register_file_->values;
          XELOGW(
              "DrawRegs draw={} prim={} idx={} DEPTHCONTROL={:08X} "
              "STENCILREFMASK={:08X} BLENDCONTROL0={:08X} COLORCONTROL={:08X} "
              "COLOR_MASK={:08X} COLOR_INFO={:08X} DEPTH_INFO={:08X} "
              "MODECONTROL={:08X} SURFACE_INFO={:08X} SU_SC_MODE_CNTL={:08X} "
              "CL_VTE_CNTL={:08X} CL_CLIP_CNTL={:08X} PROGRAM_CNTL={:08X} "
              "CONTEXT_MISC={:08X} ALPHA_REF={:08X} BLEND_RGBA={:08X},{:08X},"
              "{:08X},{:08X} VPORT_Z={:08X},{:08X} indexed={} ib_fmt={} "
              "ib_endian={} ib_count={} ib_base={:08X} ib_len={} "
              "RESET_INDX={:08X} INDX_OFFSET={:08X}",
              draw_index, uint32_t(vgt_draw_initiator.prim_type),
              vgt_draw_initiator.num_indices, rv[XE_GPU_REG_RB_DEPTHCONTROL],
              rv[XE_GPU_REG_RB_STENCILREFMASK], rv[XE_GPU_REG_RB_BLENDCONTROL0],
              rv[XE_GPU_REG_RB_COLORCONTROL], rv[XE_GPU_REG_RB_COLOR_MASK],
              rv[XE_GPU_REG_RB_COLOR_INFO], rv[XE_GPU_REG_RB_DEPTH_INFO],
              rv[XE_GPU_REG_RB_MODECONTROL], rv[XE_GPU_REG_RB_SURFACE_INFO],
              rv[XE_GPU_REG_PA_SU_SC_MODE_CNTL], rv[XE_GPU_REG_PA_CL_VTE_CNTL],
              rv[XE_GPU_REG_PA_CL_CLIP_CNTL], rv[XE_GPU_REG_SQ_PROGRAM_CNTL],
              rv[XE_GPU_REG_SQ_CONTEXT_MISC], rv[XE_GPU_REG_RB_ALPHA_REF],
              rv[XE_GPU_REG_RB_BLEND_RED], rv[XE_GPU_REG_RB_BLEND_RED + 1],
              rv[XE_GPU_REG_RB_BLEND_RED + 2], rv[XE_GPU_REG_RB_BLEND_RED + 3],
              rv[XE_GPU_REG_PA_CL_VPORT_ZSCALE], rv[XE_GPU_REG_PA_CL_VPORT_ZOFFSET],
              is_indexed ? 1 : 0, uint32_t(index_buffer_info.format),
              uint32_t(index_buffer_info.endianness), index_buffer_info.count,
              index_buffer_info.guest_base, uint64_t(index_buffer_info.length),
              rv[XE_GPU_REG_VGT_MULTI_PRIM_IB_RESET_INDX],
              rv[XE_GPU_REG_VGT_INDX_OFFSET]);
          // Vertex fetch constants 0..2 (raw) and the vf0 stream bytes
          // (/tmp/xe_vb_<draw>.bin) for CPU-side transform oracles.
          {
            // The disassembler names vertex fetch slots from the top:
            // "vf0" is fetch constant index 95 (group 31, slot 2).
            const uint32_t* fc =
                &rv[XE_GPU_REG_SHADER_CONSTANT_FETCH_00_0 + 2 * 95];
            XELOGW("DrawFetch draw={} vf0(idx95)={:08X},{:08X} "
                   "vf1(idx94)={:08X},{:08X} vf2(idx93)={:08X},{:08X}",
                   draw_index, fc[0], fc[1], fc[-2], fc[-1], fc[-4], fc[-3]);
            for (uint32_t tf = 0; tf < 6; ++tf) {
              const uint32_t* t =
                  &rv[XE_GPU_REG_SHADER_CONSTANT_FETCH_00_0 + tf * 6];
              XELOGW("DrawTexFetch draw={} tf{} raw={:08X},{:08X},{:08X},{:08X},"
                     "{:08X},{:08X} type={} base={:08X} fmt={} size={}x{} "
                     "tiled={} endian={} swz={:03X}",
                     draw_index, tf, t[0], t[1], t[2], t[3], t[4], t[5],
                     t[0] & 3u, (t[1] >> 12) << 12, (t[1] & 63u),
                     (t[2] & 0x1FFFu) + 1, ((t[2] >> 13) & 0x1FFFu) + 1,
                     (t[0] >> 31) & 1u, (t[1] >> 6) & 3u, (t[3] >> 1) & 0xFFFu);
            }
            const uint32_t vf0_addr = (fc[0] & ~3u);
            const uint32_t vf0_size = (fc[1] >> 2) & 0xFFFFFFu;  // size in dwords (bits 2-25)
            if ((fc[0] & 3u) == 3u && vf0_size) {
              const void* vb = memory_->TranslatePhysical(vf0_addr & 0x1FFFFFFF);
              if (vb) {
                std::string vb_path =
                    fmt::format("/tmp/xe_vb_{}.bin", draw_index);
                if (FILE* f = std::fopen(vb_path.c_str(), "wb")) {
                  std::fwrite(vb, 4, vf0_size, f);
                  std::fclose(f);
                }
              }
            }
          }
          // Raw guest index bytes beside the log (/tmp/xe_ib_<draw>.bin).
          if (is_indexed && index_buffer_info.length) {
            const void* ib = memory_->TranslatePhysical(
                index_buffer_info.guest_base & 0x1FFFFFFF);
            if (ib) {
              std::string ib_path =
                  fmt::format("/tmp/xe_ib_{}.bin", draw_index);
              if (FILE* f = std::fopen(ib_path.c_str(), "wb")) {
                std::fwrite(ib, 1, index_buffer_info.length, f);
                std::fclose(f);
              }
            }
          }
        }
        uint32_t force_float_saved[4] = {};
        if (s_force_float.set) {
          std::memcpy(force_float_saved,
                      &register_file_->values[s_force_float.reg],
                      sizeof(force_float_saved));
          for (uint32_t c = 0; c < 4; ++c) {
            COMMAND_PROCESSOR::WriteRegister(s_force_float.reg + c,
                                             s_force_float.words[c]);
          }
        }
        draw_succeeded = COMMAND_PROCESSOR::IssueDraw(
            vgt_draw_initiator.prim_type, vgt_draw_initiator.num_indices,
            is_indexed ? &index_buffer_info : nullptr,
            xenos::IsMajorModeExplicit(vgt_draw_initiator.major_mode,
                                       vgt_draw_initiator.prim_type));
        if (s_force_float.set) {
          for (uint32_t c = 0; c < 4; ++c) {
            COMMAND_PROCESSOR::WriteRegister(s_force_float.reg + c,
                                             force_float_saved[c]);
          }
        }
        for (uint32_t i = 0; i < s_force_reg.count; ++i) {
          if (force_reg_applied[i]) {
            COMMAND_PROCESSOR::WriteRegister(s_force_reg.reg[i],
                                             force_reg_saved[i]);
          }
        }
        if (draw_succeeded &&
            int32_t(draw_index) == cvars::replay_duplicate_draw_stencil) {
          // Diagnostic: re-issue the identical draw with zfunc=LESS, depth
          // writes off, and stencil REPLACE (ref 0xAB) on depth pass, to
          // stencil-mark any sample where the re-rasterized depth is
          // strictly below the just-written depth of the same geometry.
          uint32_t depthcontrol_value =
              register_file_->values[XE_GPU_REG_RB_DEPTHCONTROL];
          uint32_t stencilrefmask_value =
              register_file_->values[XE_GPU_REG_RB_STENCILREFMASK];
          uint32_t modecontrol_value =
              register_file_->values[XE_GPU_REG_RB_MODECONTROL];
          if (cvars::replay_duplicate_draw_depth_only) {
            reg::RB_MODECONTROL modecontrol_probe;
            modecontrol_probe.value = modecontrol_value;
            modecontrol_probe.edram_mode = xenos::EdramMode::kDepthOnly;
            register_file_->values[XE_GPU_REG_RB_MODECONTROL] =
                modecontrol_probe.value;
          }
          reg::RB_DEPTHCONTROL depthcontrol_probe;
          depthcontrol_probe.value = depthcontrol_value;
          depthcontrol_probe.z_enable = 1;
          depthcontrol_probe.z_write_enable = 0;
          depthcontrol_probe.zfunc = xenos::CompareFunction::kLess;
          depthcontrol_probe.stencil_enable = 1;
          depthcontrol_probe.backface_enable = 0;
          depthcontrol_probe.stencilfunc = xenos::CompareFunction::kAlways;
          depthcontrol_probe.stencilfail = xenos::StencilOp::kKeep;
          depthcontrol_probe.stencilzfail = xenos::StencilOp::kKeep;
          depthcontrol_probe.stencilzpass = xenos::StencilOp::kReplace;
          reg::RB_STENCILREFMASK stencilrefmask_probe;
          stencilrefmask_probe.value = 0;
          stencilrefmask_probe.stencilref = 0xAB;
          stencilrefmask_probe.stencilmask = 0xFF;
          stencilrefmask_probe.stencilwritemask = 0xFF;
          register_file_->values[XE_GPU_REG_RB_DEPTHCONTROL] =
              depthcontrol_probe.value;
          register_file_->values[XE_GPU_REG_RB_STENCILREFMASK] =
              stencilrefmask_probe.value;
          COMMAND_PROCESSOR::IssueDraw(
              vgt_draw_initiator.prim_type, vgt_draw_initiator.num_indices,
              is_indexed ? &index_buffer_info : nullptr,
              xenos::IsMajorModeExplicit(vgt_draw_initiator.major_mode,
                                         vgt_draw_initiator.prim_type));
          register_file_->values[XE_GPU_REG_RB_DEPTHCONTROL] =
              depthcontrol_value;
          register_file_->values[XE_GPU_REG_RB_STENCILREFMASK] =
              stencilrefmask_value;
          register_file_->values[XE_GPU_REG_RB_MODECONTROL] =
              modecontrol_value;
        }
        if (!draw_succeeded) {
          XELOGE("{}({}, {}, {}): Failed in backend", opcode_name,
                 vgt_draw_initiator.num_indices,
                 uint32_t(vgt_draw_initiator.prim_type),
                 uint32_t(vgt_draw_initiator.source_select));
        }
      }
      // Selective EDRAM capture AFTER this draw issued (design D3, §2):
      // writes <capture_dir>/draw_NNNNNN.edram.bin for draws chosen by
      // --capture_draw_range / --capture_edram_at, and/or appends the
      // per-tile/sample-plane hash record when --capture_edram_hashes. May
      // perturb (forces a submission + queue drain); no-op when not selected.
      COMMAND_PROCESSOR::MaybeCaptureEdramSnapshot(draw_index);
      // Hazard-bisection probe (--serialize_draw_range): submission + wait
      // after selected draws, no capture output.
      COMMAND_PROCESSOR::MaybeSerializeDraw(draw_index);
    }
  }

  // If read the packed correctly, but merely couldn't execute it (because of,
  // for instance, features not supported by the host), don't terminate command
  // buffer processing as that would leave rendering in a way more inconsistent
  // state than just a single dropped draw command.
  return true;
}

bool COMMAND_PROCESSOR::ExecutePacketType3_DRAW_INDX(
    uint32_t packet, uint32_t count) XE_RESTRICT {
  // "initiate fetch of index buffer and draw"
  // Generally used by Xbox 360 Direct3D 9 for kDMA and kAutoIndex sources.
  // With a viz query token as the first one.
  uint32_t count_remaining = count;
  assert_not_zero(count_remaining);
  if (!count_remaining) {
    XELOGE("PM4_DRAW_INDX: Packet too small, can't read the viz query token");
    return false;
  }
  uint32_t viz_query_condition = reader_.ReadAndSwap<uint32_t>();
  --count_remaining;
  return COMMAND_PROCESSOR::ExecutePacketType3Draw(
      packet, "PM4_DRAW_INDX", viz_query_condition, count_remaining);
}

bool COMMAND_PROCESSOR::ExecutePacketType3_DRAW_INDX_2(
    uint32_t packet, uint32_t count) XE_RESTRICT {
  // "draw using supplied indices in packet"
  // Generally used by Xbox 360 Direct3D 9 for kAutoIndex source.
  // No viz query token.
  return COMMAND_PROCESSOR::ExecutePacketType3Draw(packet, "PM4_DRAW_INDX_2", 0,
                                                   count);
}
XE_FORCEINLINE
bool COMMAND_PROCESSOR::ExecutePacketType3_SET_CONSTANT(
    uint32_t packet, uint32_t count) XE_RESTRICT {
  // load constant into chip and to memory
  // PM4_REG(reg) ((0x4 << 16) | (GSL_HAL_SUBBLOCK_OFFSET(reg)))
  //                                     reg - 0x2000
  uint32_t offset_type = reader_.ReadAndSwap<uint32_t>();
  uint32_t index = offset_type & 0x7FF;
  uint32_t type = (offset_type >> 16) & 0xFF;
  uint32_t countm1 = count - 1;
  switch (type) {
    case 0:  // ALU
      // index += 0x4000;
      // COMMAND_PROCESSOR::WriteRegisterRangeFromRing( index, countm1);
      COMMAND_PROCESSOR::WriteALURangeFromRing(&reader_, index, countm1);
      break;
    case 1:  // FETCH

      COMMAND_PROCESSOR::WriteFetchRangeFromRing(&reader_, index, countm1);

      break;
    case 2:  // BOOL
      COMMAND_PROCESSOR::WriteBoolRangeFromRing(&reader_, index, countm1);

      break;
    case 3:  // LOOP

      COMMAND_PROCESSOR::WriteLoopRangeFromRing(&reader_, index, countm1);

      break;
    case 4:  // REGISTERS

      COMMAND_PROCESSOR::WriteREGISTERSRangeFromRing(&reader_, index, countm1);

      break;
    default:
      assert_always();
      reader_.AdvanceRead((count - 1) * sizeof(uint32_t));
      return true;
  }

  return true;
}
XE_NOINLINE
bool COMMAND_PROCESSOR::ExecutePacketType3_SET_CONSTANT2(
    uint32_t packet, uint32_t count) XE_RESTRICT {
  uint32_t offset_type = reader_.ReadAndSwap<uint32_t>();
  uint32_t index = offset_type & 0xFFFF;
  uint32_t countm1 = count - 1;

  COMMAND_PROCESSOR::WriteRegisterRangeFromRing(&reader_, index, countm1);

  return true;
}
XE_FORCEINLINE
bool COMMAND_PROCESSOR::ExecutePacketType3_LOAD_ALU_CONSTANT(
    uint32_t packet, uint32_t count) XE_RESTRICT {
  // load constants from memory
  uint32_t address = reader_.ReadAndSwap<uint32_t>();
  address &= 0x3FFFFFFF;
  uint32_t offset_type = reader_.ReadAndSwap<uint32_t>();
  uint32_t index = offset_type & 0x7FF;
  uint32_t size_dwords = reader_.ReadAndSwap<uint32_t>();
  size_dwords &= 0xFFF;
  uint32_t type = (offset_type >> 16) & 0xFF;

  // The packet's 0x3FFFFFFF mask above leaves the type/flag bits already
  // stripped; resolve the single 512 MiB physical alias (matching
  // Memory::TranslatePhysical) used for the authority boundary, the trace
  // reads, and the actual RAM read.
  uint32_t constant_physical = xenos::CpuToGpu(address);
  if (!COMMAND_PROCESSOR::PrepareGuestMemoryForHostRead(
          constant_physical, size_dwords * sizeof(uint32_t))) {
    XELOGE("LOAD_ALU_CONSTANT: failed to prepare guest range {:08X}(+{})",
           constant_physical, size_dwords * sizeof(uint32_t));
    return false;
  }

  auto xlat_address = (uint32_t*)memory_->TranslatePhysical(constant_physical);

  switch (type) {
    case 0:  // ALU
      trace_writer_.WriteMemoryRead(constant_physical, size_dwords * 4);
      COMMAND_PROCESSOR::WriteALURangeFromMem(index, xlat_address, size_dwords);

      break;
    case 1:  // FETCH
      trace_writer_.WriteMemoryRead(constant_physical, size_dwords * 4);
      COMMAND_PROCESSOR::WriteFetchRangeFromMem(index, xlat_address,
                                                size_dwords);
      break;
    case 2:  // BOOL
      trace_writer_.WriteMemoryRead(constant_physical, size_dwords * 4);

      COMMAND_PROCESSOR::WriteBoolRangeFromMem(index, xlat_address,
                                               size_dwords);
      break;
    case 3:  // LOOP
      trace_writer_.WriteMemoryRead(constant_physical, size_dwords * 4);

      COMMAND_PROCESSOR::WriteLoopRangeFromMem(index, xlat_address,
                                               size_dwords);

      break;
    case 4:  // REGISTERS
      // chrispy: todo, REGISTERS cannot write any special regs, so optimize for
      // that
      trace_writer_.WriteMemoryRead(constant_physical, size_dwords * 4);

      COMMAND_PROCESSOR::WriteREGISTERSRangeFromMem(index, xlat_address,
                                                    size_dwords);
      break;
    default:
      assert_always();
      return true;
  }

  return true;
}

bool COMMAND_PROCESSOR::ExecutePacketType3_SET_SHADER_CONSTANTS(
    uint32_t packet, uint32_t count) XE_RESTRICT {
  uint32_t offset_type = reader_.ReadAndSwap<uint32_t>();
  uint32_t index = offset_type & 0xFFFF;
  uint32_t countm1 = count - 1;
  COMMAND_PROCESSOR::WriteRegisterRangeFromRing(&reader_, index, countm1);

  return true;
}

bool COMMAND_PROCESSOR::ExecutePacketType3_IM_LOAD(uint32_t packet,
                                                   uint32_t count) XE_RESTRICT {
  SCOPE_profile_cpu_f("gpu");

  // load sequencer instruction memory (pointer-based)
  uint32_t addr_type = reader_.ReadAndSwap<uint32_t>();
  auto shader_type = static_cast<xenos::ShaderType>(addr_type & 0x3);
  uint32_t addr = addr_type & ~0x3;
  uint32_t start_size = reader_.ReadAndSwap<uint32_t>();
  uint32_t start = start_size >> 16;
  uint32_t size_dwords = start_size & 0xFFFF;  // dwords
  assert_true(start == 0);
  // addr already has the packet's low shader-type bits stripped. Resolve the
  // single 512 MiB physical alias (matching Memory::TranslatePhysical) used for
  // the authority boundary, trace read, shader identity, and actual RAM read.
  uint32_t shader_physical = xenos::CpuToGpu(addr);
  if (!COMMAND_PROCESSOR::PrepareGuestMemoryForHostRead(
          shader_physical, size_dwords * sizeof(uint32_t))) {
    XELOGE("IM_LOAD: failed to prepare guest shader {:08X}(+{})", shader_physical,
           size_dwords * sizeof(uint32_t));
    return false;
  }
  trace_writer_.WriteMemoryRead(shader_physical, size_dwords * 4);
  auto shader = COMMAND_PROCESSOR::LoadShader(
      shader_type, shader_physical,
      memory_->TranslatePhysical<uint32_t*>(shader_physical), size_dwords);
  switch (shader_type) {
    case xenos::ShaderType::kVertex:
      active_vertex_shader_ = shader;
      break;
    case xenos::ShaderType::kPixel:
      active_pixel_shader_ = shader;
      break;
    default:
      assert_unhandled_case(shader_type);
      return false;
  }
  return true;
}

bool COMMAND_PROCESSOR::ExecutePacketType3_IM_LOAD_IMMEDIATE(
    uint32_t packet, uint32_t count) XE_RESTRICT {
  SCOPE_profile_cpu_f("gpu");

  // load sequencer instruction memory (code embedded in packet)
  uint32_t dword0 = reader_.ReadAndSwap<uint32_t>();
  uint32_t dword1 = reader_.ReadAndSwap<uint32_t>();
  auto shader_type = static_cast<xenos::ShaderType>(dword0);
  uint32_t start_size = dword1;
  uint32_t start = start_size >> 16;
  uint32_t size_dwords = start_size & 0xFFFF;  // dwords
  assert_true(start == 0);
  assert_true(reader_.read_count() >= size_dwords * 4);
  assert_true(count - 2 >= size_dwords);
  auto shader = COMMAND_PROCESSOR::LoadShader(
      shader_type, uint32_t(reader_.read_ptr()),
      reinterpret_cast<uint32_t*>(reader_.read_ptr()), size_dwords);
  switch (shader_type) {
    case xenos::ShaderType::kVertex:
      active_vertex_shader_ = shader;
      break;
    case xenos::ShaderType::kPixel:
      active_pixel_shader_ = shader;
      break;
    default:
      assert_unhandled_case(shader_type);
      return false;
  }
  reader_.AdvanceRead(size_dwords * sizeof(uint32_t));
  return true;
}

/*
        todo: shouldn't this do something?
*/

bool COMMAND_PROCESSOR::ExecutePacketType3_INVALIDATE_STATE(
    uint32_t packet, uint32_t count) XE_RESTRICT {
  // selective invalidation of state pointers
  /*uint32_t mask =*/reader_.ReadAndSwap<uint32_t>();
  // driver_->InvalidateState(mask);
  return true;
}

bool COMMAND_PROCESSOR::ExecutePacketType3_VIZ_QUERY(
    uint32_t packet, uint32_t count) XE_RESTRICT {
  // begin/end initiator for viz query extent processing
  // https://www.google.com/patents/US20050195186
  assert_true(count == 1);

  uint32_t dword0 = reader_.ReadAndSwap<uint32_t>();

  uint32_t id = dword0 & 0x3F;
  uint32_t end = dword0 & 0x100;
  {
    // Upstream-delta census (XE_GPU_DELTA_CENSUS, 2026-10-09): VIZ_QUERY
    // begin/end packets, which Canary 692cd59cf turns into visibility surveys.
    static const bool s_viz_census =
        std::getenv("XE_GPU_DELTA_CENSUS") != nullptr;
    static uint32_t s_viz_census_lines = 0;
    if (s_viz_census && s_viz_census_lines < 2000) {
      ++s_viz_census_lines;
      XELOGI("DeltaCensus viz_packet id={} end={}", dword0 & 0x3F,
             (dword0 & 0x100) ? 1 : 0);
    }
  }
  if (!end) {
    // begin a new viz query @ id
    // On hardware this clears the internal state of the scan converter (which
    // is different to the register)
    COMMAND_PROCESSOR::WriteEventInitiator(VIZQUERY_START);
    // XELOGGPU("Begin viz query ID {:02X}", id);
  } else {
    // end the viz query
    COMMAND_PROCESSOR::WriteEventInitiator(VIZQUERY_END);
    // XELOGGPU("End viz query ID {:02X}", id);
    // The scan converter writes the internal result back to the register here.
    // We just fake it and say it was visible in case it is read back.
    if (id < 32) {
      register_file_->values[XE_GPU_REG_PA_SC_VIZ_QUERY_STATUS_0] |= uint32_t(1)
                                                                     << id;
    } else {
      register_file_->values[XE_GPU_REG_PA_SC_VIZ_QUERY_STATUS_1] |=
          uint32_t(1) << (id - 32);
    }
  }

  return true;
}

uint32_t COMMAND_PROCESSOR::ExecutePrimaryBuffer(uint32_t read_index,
                                                 uint32_t write_index) {
  SCOPE_profile_cpu_f("gpu");
#if XE_ENABLE_TRACE_WRITER_INSTRUMENTATION == 1
  // If we have a pending trace stream open it now. That way we ensure we get
  // all commands.
  if (!trace_writer_.is_open() && trace_state_ == TraceState::kStreaming) {
    uint32_t title_id = kernel_state_->GetExecutableModule()
                            ? kernel_state_->GetExecutableModule()->title_id()
                            : 0;
    auto file_name = BuildTraceFileName(title_id, counter_, true);
    auto path = trace_stream_path_ / file_name;
    if (OpenTrace(path, title_id, counter_, true)) {
      InitializeTrace();
    }
  }
#endif
  // Adjust pointer base.
  uint32_t start_ptr = primary_buffer_ptr_ + read_index * sizeof(uint32_t);
  start_ptr = (primary_buffer_ptr_ & ~0x1FFFFFFF) | (start_ptr & 0x1FFFFFFF);
  uint32_t end_ptr = primary_buffer_ptr_ + write_index * sizeof(uint32_t);
  end_ptr = (primary_buffer_ptr_ & ~0x1FFFFFFF) | (end_ptr & 0x1FFFFFFF);

  trace_writer_.WritePrimaryBufferStart(start_ptr, write_index - read_index);

  // Execute commands!

  RingBuffer old_reader = reader_;
  new (&reader_) RingBuffer(memory_->TranslatePhysical(primary_buffer_ptr_),
                            primary_buffer_size_);

  reader_.set_read_offset(read_index * sizeof(uint32_t));
  reader_.set_write_offset(write_index * sizeof(uint32_t));
  // prefetch the wraparound range
  // it likely is already in L3 cache, but in a zen system it may be another
  // chiplets l3
  reader_.BeginPrefetchedRead<swcache::PrefetchTag::Level2>(
      GetCurrentRingReadCount());
  do {
    if (!COMMAND_PROCESSOR::ExecutePacket()) {
      // This probably should be fatal - but we're going to continue anyways.
      XELOGE("**** PRIMARY RINGBUFFER: Failed to execute packet.");
      assert_always();
      break;
    }
  } while (reader_.read_count());

  COMMAND_PROCESSOR::OnPrimaryBufferEnd();

  trace_writer_.WritePrimaryBufferEnd();

  reader_ = old_reader;
  return write_index;
}

void COMMAND_PROCESSOR::ExecutePacket(uint32_t ptr, uint32_t count) {
  // Execute commands!
  RingBuffer old_reader = reader_;

  new (&reader_)
      RingBuffer{memory_->TranslatePhysical(ptr), count * sizeof(uint32_t)};

  reader_.set_write_offset(count * sizeof(uint32_t));

  do {
    if (!COMMAND_PROCESSOR::ExecutePacket()) {
      XELOGE("**** ExecutePacket: Failed to execute packet.");
      assert_always();
      break;
    }
  } while (reader_.read_count());
  reader_ = old_reader;
}
