import { useEffect, useRef } from "react";
import { useTheme, useTone } from "./theme";

const vertexShader = `
attribute vec2 position;
void main() {
  gl_Position = vec4(position, 0.0, 1.0);
}
`;

/**
 * The field the blades sit on: brushed grey with slow drifting light, washed by
 * whichever blade is currently open. It shows at the deck's outer edges and
 * between the blade shoulders, so it stays deliberately quiet.
 */
const fragmentShader = `
precision mediump float;
uniform vec2 resolution;
uniform float time;
uniform float dark;
uniform vec3 tone;

float hash(vec2 p) {
  return fract(sin(dot(p, vec2(41.7, 289.3))) * 43758.5453);
}

void main() {
  vec2 uv = gl_FragCoord.xy / resolution.xy;

  // Mirror about the centre: the sheets stack the same way on both sides.
  float x = min(uv.x, 1.0 - uv.x);

  // The sheets bow inward at mid-height and splay out toward top and bottom.
  float bow = 1.0 - pow(abs(uv.y - 0.5) * 2.0, 1.7);
  float e0 = 0.026 + 0.026 * bow;
  float e1 = 0.062 + 0.038 * bow;
  float e2 = 0.104 + 0.050 * bow;

  vec3 pale = vec3(0.878, 0.867, 0.886);
  vec3 mid = vec3(0.756, 0.745, 0.768);
  vec3 bright = vec3(0.941, 0.933, 0.949);

  // Overlapping curved sheets of brushed metal.
  vec3 col = pale;
  col = mix(col, mid, smoothstep(e0 - 0.003, e0 + 0.003, x));
  col = mix(col, bright, smoothstep(e1 - 0.003, e1 + 0.003, x));
  col = mix(col, mix(pale, mid, 0.5), smoothstep(e2 - 0.003, e2 + 0.003, x));

  // A specular catch just inside each seam.
  col += 0.10 * exp(-pow((x - e0) / 0.006, 2.0));
  col += 0.12 * exp(-pow((x - e1) / 0.006, 2.0));
  col += 0.08 * exp(-pow((x - e2) / 0.006, 2.0));

  // Soft shadow where one sheet passes under the next.
  col -= 0.09 * exp(-pow((x - e0 - 0.012) / 0.010, 2.0));
  col -= 0.09 * exp(-pow((x - e1 - 0.012) / 0.010, 2.0));

  // Brushed grain running down the sheets.
  col += (hash(vec2(floor(gl_FragCoord.x), 0.0)) - 0.5) * 0.035;

  // The open blade tints the metal it sits against.
  float wash = smoothstep(0.34, 0.02, x) * 0.0 + smoothstep(0.18, 0.5, x) * 0.5;
  col = mix(col, tone, wash * 0.18);

  // Night: the same sheets, dimmed rather than recoloured.
  vec3 night = col * 0.11 + tone * 0.10;
  col = mix(col, night, dark);

  float vig = smoothstep(1.35, 0.35, length((uv - 0.5) * vec2(1.1, 1.0)));
  col *= mix(0.88, 1.0, vig);
  col += (hash(gl_FragCoord.xy) - 0.5) * 0.006;

  gl_FragColor = vec4(col, 1.0);
}
`;

const TONES: Record<string, [number, number, number]> = {
  green: [0.373, 0.733, 0.306],
  orange: [0.961, 0.569, 0.243],
  amber: [0.878, 0.635, 0.184],
  blue: [0.255, 0.608, 0.886],
  purple: [0.643, 0.498, 0.871],
  steel: [0.553, 0.604, 0.659],
  teal: [0.247, 0.702, 0.639],
  rose: [0.851, 0.376, 0.498]
};

const TONES_DARK: Record<string, [number, number, number]> = {
  green: [0.11, 0.28, 0.07],
  orange: [0.38, 0.2, 0.06],
  amber: [0.36, 0.24, 0.04],
  blue: [0.09, 0.24, 0.36],
  purple: [0.2, 0.15, 0.3],
  steel: [0.16, 0.19, 0.22],
  teal: [0.06, 0.24, 0.21],
  rose: [0.25, 0.08, 0.14]
};

function compileShader(gl: WebGLRenderingContext, type: number, source: string) {
  const shader = gl.createShader(type);
  if (!shader) return null;
  gl.shaderSource(shader, source);
  gl.compileShader(shader);
  return gl.getShaderParameter(shader, gl.COMPILE_STATUS) ? shader : null;
}

export function DashboardField() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const targetRef = useRef({ dark: 0, tone: TONES.green });
  const theme = useTheme();
  const tone = useTone();

  useEffect(() => {
    const dark = theme === "dark";
    const table = dark ? TONES_DARK : TONES;
    targetRef.current = {
      dark: dark ? 1 : 0,
      tone: table[tone] ?? table.green
    };
  }, [theme, tone]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const gl = canvas.getContext("webgl", {
      antialias: false,
      alpha: false,
      powerPreference: "low-power"
    });
    if (!gl) return;

    const vert = compileShader(gl, gl.VERTEX_SHADER, vertexShader);
    const frag = compileShader(gl, gl.FRAGMENT_SHADER, fragmentShader);
    const program = gl.createProgram();
    if (!vert || !frag || !program) return;

    gl.attachShader(program, vert);
    gl.attachShader(program, frag);
    gl.linkProgram(program);
    if (!gl.getProgramParameter(program, gl.LINK_STATUS)) return;
    gl.useProgram(program);

    const buffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
    gl.bufferData(
      gl.ARRAY_BUFFER,
      new Float32Array([-1, -1, 1, -1, -1, 1, -1, 1, 1, -1, 1, 1]),
      gl.STATIC_DRAW
    );

    const position = gl.getAttribLocation(program, "position");
    gl.enableVertexAttribArray(position);
    gl.vertexAttribPointer(position, 2, gl.FLOAT, false, 0, 0);

    const uResolution = gl.getUniformLocation(program, "resolution");
    const uTime = gl.getUniformLocation(program, "time");
    const uDark = gl.getUniformLocation(program, "dark");
    const uTone = gl.getUniformLocation(program, "tone");

    const reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    // Phones show brushed metal instead (globals.css); don't draw a hidden
    // field every frame.
    const narrow = window.matchMedia("(max-width: 860px)");
    let frame = 0;
    const eased = { dark: targetRef.current.dark, tone: [...targetRef.current.tone] };

    const resize = () => {
      // A soft gradient needs no detail; half resolution keeps this off the
      // critical path on integrated GPUs.
      const ratio = Math.min(window.devicePixelRatio || 1, 1.25);
      canvas.width = Math.max(1, Math.floor(window.innerWidth * ratio));
      canvas.height = Math.max(1, Math.floor(window.innerHeight * ratio));
      gl.viewport(0, 0, canvas.width, canvas.height);
    };

    const draw = (now: number) => {
      const target = targetRef.current;
      const root = document.documentElement.dataset;
      const still =
        root.motion === "reduced" || (reduced && root.motionOverride !== "on");
      const k = still ? 1 : 0.06;
      eased.dark += (target.dark - eased.dark) * k;
      for (let i = 0; i < 3; i += 1) {
        eased.tone[i] += (target.tone[i] - eased.tone[i]) * k;
      }

      gl.uniform2f(uResolution, canvas.width, canvas.height);
      gl.uniform1f(uTime, reduced ? 0 : now * 0.001);
      gl.uniform1f(uDark, eased.dark);
      gl.uniform3f(uTone, eased.tone[0], eased.tone[1], eased.tone[2]);
      gl.drawArrays(gl.TRIANGLES, 0, 6);
      frame = narrow.matches ? 0 : requestAnimationFrame(draw);
    };

    const onWidth = () => {
      if (!narrow.matches && !frame) frame = requestAnimationFrame(draw);
    };

    resize();
    window.addEventListener("resize", resize);
    narrow.addEventListener("change", onWidth);
    frame = narrow.matches ? 0 : requestAnimationFrame(draw);

    return () => {
      cancelAnimationFrame(frame);
      window.removeEventListener("resize", resize);
      narrow.removeEventListener("change", onWidth);
      gl.deleteBuffer(buffer);
      gl.deleteProgram(program);
      gl.deleteShader(vert);
      gl.deleteShader(frag);
    };
  }, []);

  return <canvas ref={canvasRef} className="dash-field" aria-hidden="true" />;
}
