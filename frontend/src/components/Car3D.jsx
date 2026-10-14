import { useEffect, useMemo, useRef, useState } from "react";
import * as THREE from "three";
import { Canvas, useFrame, useThree } from "@react-three/fiber";
import { Html, OrbitControls } from "@react-three/drei";
import { COMP_ORDER, COMP_SHORT, STATUS_COLOR, STATUS_LABEL, signed } from "../lib.js";

// Hotspots and label offsets for the procedural sedan (front of the car is +x).
const HOT = {
  engine: { pos: [1.35, 0.78, 0.0], off: [0.0, 0.85, 0.2] },
  cooling: { pos: [2.12, 0.62, 0.0], off: [0.75, 0.25, 0.55] },
  lubrication: { pos: [1.2, 0.36, 0.0], off: [0.2, -0.1, 1.15] },
  brakes: { pos: [1.35, 0.4, 0.98], off: [0.9, 0.0, 0.7] },
  battery: { pos: [1.72, 0.76, -0.55], off: [0.55, 0.95, -0.9] },
  general_wear: { pos: [-1.5, 0.75, 0.0], off: [-0.4, 0.95, 0.0] },
};

const DEFAULT_VIEW = { pos: [6.0, 3.0, 6.3], target: [0.75, 0.55, 0] };
const FOCUS = {
  engine: { pos: [3.7, 2.1, 2.7], target: [1.4, 0.8, 0] },
  cooling: { pos: [4.4, 1.3, 2.3], target: [2.0, 0.6, 0] },
  lubrication: { pos: [3.2, 0.6, 2.7], target: [1.2, 0.35, 0] },
  brakes: { pos: [3.4, 1.0, 3.0], target: [1.35, 0.4, 0.9] },
  battery: { pos: [3.6, 1.9, -2.5], target: [1.7, 0.75, -0.5] },
  general_wear: { pos: [-3.8, 2.4, 3.6], target: [-0.8, 0.7, 0] },
};

function sedanShape() {
  const s = new THREE.Shape();
  s.moveTo(-2.15, 0.3);
  s.lineTo(-2.2, 0.75);
  s.quadraticCurveTo(-2.2, 0.93, -1.9, 0.96);
  s.lineTo(-1.5, 0.99);
  s.quadraticCurveTo(-1.0, 1.05, -0.65, 1.4);
  s.quadraticCurveTo(-0.5, 1.53, -0.2, 1.53);
  s.lineTo(0.35, 1.51);
  s.quadraticCurveTo(0.65, 1.48, 0.95, 1.1);
  s.quadraticCurveTo(1.1, 0.99, 1.4, 0.97);
  s.lineTo(1.95, 0.89);
  s.quadraticCurveTo(2.2, 0.83, 2.22, 0.55);
  s.lineTo(2.2, 0.3);
  s.lineTo(-2.15, 0.3);
  return s;
}

function glassShape() {
  const g = new THREE.Shape();
  g.moveTo(-1.3, 1.03);
  g.quadraticCurveTo(-0.95, 1.08, -0.6, 1.4);
  g.quadraticCurveTo(-0.48, 1.46, -0.2, 1.46);
  g.lineTo(0.35, 1.44);
  g.quadraticCurveTo(0.6, 1.42, 0.88, 1.08);
  g.lineTo(0.88, 1.03);
  g.lineTo(-1.3, 1.03);
  return g;
}

function CarBody({ xray }) {
  const body = useMemo(() => {
    const geo = new THREE.ExtrudeGeometry(sedanShape(), {
      depth: 1.6, bevelEnabled: true, bevelThickness: 0.1, bevelSize: 0.1, bevelSegments: 4, curveSegments: 18,
    });
    geo.translate(0, 0, -0.8);
    return geo;
  }, []);
  const outline = useMemo(
    () => new THREE.LineSegments(
      new THREE.EdgesGeometry(body, 22),
      new THREE.LineBasicMaterial({ color: "#38bdf8", transparent: true, opacity: 0.55 })
    ),
    [body]
  );
  const glass = useMemo(() => {
    const geo = new THREE.ExtrudeGeometry(glassShape(), { depth: 1.84, bevelEnabled: false, curveSegments: 14 });
    geo.translate(0, 0, -0.92);
    return geo;
  }, []);
  return (
    <group>
      <mesh geometry={body}>
        <meshStandardMaterial
          color={xray ? "#2b6cb0" : "#86aee0"} metalness={0.4} roughness={0.38}
          transparent opacity={xray ? 0.09 : 1} depthWrite={!xray}
        />
      </mesh>
      {xray && <primitive object={outline} />}
      <mesh geometry={glass}>
        <meshStandardMaterial color="#0a1a33" metalness={0.8} roughness={0.1} transparent opacity={xray ? 0.12 : 0.85} depthWrite={!xray} />
      </mesh>
      {[[2.24, 0.68, -0.62], [2.24, 0.68, 0.62]].map((p, i) => (
        <mesh key={`h${i}`} position={p}>
          <boxGeometry args={[0.05, 0.1, 0.4]} />
          <meshStandardMaterial color="#fff6d6" emissive="#fff0b0" emissiveIntensity={xray ? 0.2 : 0.9} />
        </mesh>
      ))}
      {[[-2.22, 0.75, -0.62], [-2.22, 0.75, 0.62]].map((p, i) => (
        <mesh key={`t${i}`} position={p}>
          <boxGeometry args={[0.05, 0.1, 0.4]} />
          <meshStandardMaterial color="#7a1c1c" emissive="#ff3030" emissiveIntensity={xray ? 0.2 : 0.7} />
        </mesh>
      ))}
      {[[1.35, 0.4, 0.97], [1.35, 0.4, -0.97], [-1.3, 0.4, 0.97], [-1.3, 0.4, -0.97]].map((p, i) => (
        <group key={i} position={p}>
          <mesh rotation={[Math.PI / 2, 0, 0]}>
            <cylinderGeometry args={[0.42, 0.42, 0.28, 32]} />
            <meshStandardMaterial color="#0b0f16" roughness={0.9} transparent opacity={xray ? 0.35 : 1} />
          </mesh>
          <mesh rotation={[Math.PI / 2, 0, 0]} position={[0, 0, p[2] > 0 ? 0.02 : -0.02]}>
            <cylinderGeometry args={[0.26, 0.26, 0.3, 24]} />
            <meshStandardMaterial color="#9aa8bd" metalness={0.8} roughness={0.3} transparent opacity={xray ? 0.35 : 1} />
          </mesh>
        </group>
      ))}
    </group>
  );
}

// Status-coloured internals shown in X-ray mode. Each group is clickable.
function Internals({ statusOf, selected, onSelect, show }) {
  if (!show) return null;
  const mat = (id) => {
    const c = STATUS_COLOR[statusOf(id)] || STATUS_COLOR.idle;
    const sel = selected === id;
    return <meshStandardMaterial color={c} emissive={c} emissiveIntensity={sel ? 1.1 : 0.55} metalness={0.3} roughness={0.4} />;
  };
  const click = (id) => (e) => { e.stopPropagation(); onSelect(id); };
  const pointer = {
    onPointerOver: () => (document.body.style.cursor = "pointer"),
    onPointerOut: () => (document.body.style.cursor = "auto"),
  };
  return (
    <group>
      <group position={[1.35, 0.7, 0]} onClick={click("engine")} {...pointer} scale={selected === "engine" ? 1.08 : 1}>
        <mesh><boxGeometry args={[0.95, 0.42, 0.62]} />{mat("engine")}</mesh>
        {[-0.33, -0.11, 0.11, 0.33].map((x) => (
          <mesh key={x} position={[x, 0.3, 0]}><cylinderGeometry args={[0.08, 0.08, 0.22, 14]} />{mat("engine")}</mesh>
        ))}
      </group>
      <group position={[2.05, 0.62, 0]} onClick={click("cooling")} {...pointer} scale={selected === "cooling" ? 1.1 : 1}>
        <mesh><boxGeometry args={[0.09, 0.5, 1.1]} />{mat("cooling")}</mesh>
        <mesh position={[-0.12, 0, 0]} rotation={[0, 0, Math.PI / 2]}><cylinderGeometry args={[0.2, 0.2, 0.05, 20]} />{mat("cooling")}</mesh>
      </group>
      <group position={[1.2, 0.36, 0]} onClick={click("lubrication")} {...pointer} scale={selected === "lubrication" ? 1.1 : 1}>
        <mesh><boxGeometry args={[0.8, 0.12, 0.5]} />{mat("lubrication")}</mesh>
      </group>
      <group position={[1.72, 0.76, -0.55]} onClick={click("battery")} {...pointer} scale={selected === "battery" ? 1.15 : 1}>
        <mesh><boxGeometry args={[0.36, 0.24, 0.26]} />{mat("battery")}</mesh>
      </group>
      {[[1.35, 0.4, 0.82], [1.35, 0.4, -0.82], [-1.3, 0.4, 0.82], [-1.3, 0.4, -0.82]].map((p, i) => (
        <group key={i} position={p} onClick={click("brakes")} {...pointer} scale={selected === "brakes" ? 1.1 : 1}>
          <mesh rotation={[Math.PI / 2, 0, 0]}><cylinderGeometry args={[0.3, 0.3, 0.05, 28]} />{mat("brakes")}</mesh>
          <mesh position={[0.28, 0.08, 0]}><boxGeometry args={[0.14, 0.18, 0.12]} />{mat("brakes")}</mesh>
        </group>
      ))}
      <group onClick={click("general_wear")} {...pointer} scale={selected === "general_wear" ? 1.03 : 1}>
        {[-0.55, 0.55].map((z) => (
          <mesh key={z} position={[0, 0.34, z]}><boxGeometry args={[4.1, 0.06, 0.07]} />{mat("general_wear")}</mesh>
        ))}
        {[-1.4, 0, 1.0].map((x) => (
          <mesh key={x} position={[x, 0.34, 0]}><boxGeometry args={[0.07, 0.06, 1.1]} />{mat("general_wear")}</mesh>
        ))}
      </group>
    </group>
  );
}

function Hotspot({ id, status, selected, onSelect, showLabel, detail, hideDot }) {
  const meta = HOT[id];
  const color = STATUS_COLOR[status] || STATUS_COLOR.idle;
  const ring = useRef();
  const [hover, setHover] = useState(false);
  useFrame(({ clock }) => {
    if (!ring.current) return;
    const t = (clock.getElapsedTime() * (status === "critical" ? 1.6 : 0.9)) % 1;
    const calm = status === "ok" || status === "idle";
    const s = 1 + t * (calm ? 0.5 : 1.4);
    ring.current.scale.set(s, s, s);
    ring.current.material.opacity = (1 - t) * (calm ? 0.25 : 0.6);
  });
  const line = useMemo(() => {
    const geo = new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(0, 0, 0), new THREE.Vector3(...meta.off)]);
    const mat = new THREE.LineBasicMaterial({ color, transparent: true, opacity: 0.7, depthTest: false });
    return new THREE.Line(geo, mat);
  }, [meta, color]);
  return (
    <group position={meta.pos}>
      {!hideDot && (
        <>
          <mesh
            renderOrder={10}
            scale={selected || hover ? 1.25 : 1}
            onClick={(e) => { e.stopPropagation(); onSelect(id); }}
            onPointerOver={(e) => { e.stopPropagation(); setHover(true); document.body.style.cursor = "pointer"; }}
            onPointerOut={() => { setHover(false); document.body.style.cursor = "auto"; }}
          >
            <sphereGeometry args={[0.14, 24, 24]} />
            <meshStandardMaterial color={color} emissive={color} emissiveIntensity={0.9} depthTest={false} transparent />
          </mesh>
          <mesh ref={ring} renderOrder={9}>
            <sphereGeometry args={[0.14, 20, 20]} />
            <meshBasicMaterial color={color} transparent opacity={0.4} depthTest={false} depthWrite={false} />
          </mesh>
        </>
      )}
      {(showLabel || hover) && (
        <>
          <primitive object={line} />
          <Html position={meta.off} center zIndexRange={[20, 0]}>
            <div className="hot-label" style={{ borderColor: color, color }} onClick={() => onSelect(id)}>
              <b>{COMP_SHORT[id]}</b>
              <span>{detail}</span>
            </div>
          </Html>
        </>
      )}
    </group>
  );
}

function Rig({ focus, controls }) {
  const { camera } = useThree();
  const goal = useRef(DEFAULT_VIEW);
  const left = useRef(0);
  useEffect(() => {
    goal.current = FOCUS[focus] || DEFAULT_VIEW;
    left.current = 1.4;
  }, [focus]);
  useFrame((_, dt) => {
    const c = controls.current;
    if (!c || left.current <= 0) return;
    const k = 1 - Math.pow(0.002, dt);
    const tgt = new THREE.Vector3(...goal.current.target);
    const pos = new THREE.Vector3(...goal.current.pos);
    if (goal.current !== DEFAULT_VIEW) pos.sub(tgt).multiplyScalar(1.3).add(tgt); // keep some of the car in frame
    camera.position.lerp(pos, k);
    c.target.lerp(tgt, k);
    c.update();
    left.current -= dt;
  });
  return null;
}

function Floor() {
  return (
    <group>
      <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, 0.005, 0]}>
        <ringGeometry args={[3.35, 3.45, 96]} />
        <meshBasicMaterial color="#3b9bff" />
      </mesh>
      <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, 0.004, 0]}>
        <ringGeometry args={[3.45, 3.9, 96]} />
        <meshBasicMaterial color="#3b9bff" transparent opacity={0.12} />
      </mesh>
      <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, 0.002, 0]}>
        <circleGeometry args={[3.35, 64]} />
        <meshBasicMaterial color="#0f2a52" transparent opacity={0.55} />
      </mesh>
      <gridHelper args={[16, 32, "#1d3a66", "#10233f"]} position={[0, -0.001, 0]} />
    </group>
  );
}

export default function Car3D({
  components, selected, onSelect, mode = "exterior", focus = null, labels = "all", height = 460,
  autoRotate = false, legend = true,
}) {
  const controls = useRef();
  const byId = (id) => components?.find((c) => c.id === id);
  const statusOf = (id) => byId(id)?.status || "idle";
  const xray = mode === "xray";
  return (
    <div style={{ height, position: "relative" }}>
      <Canvas camera={{ position: DEFAULT_VIEW.pos, fov: 38 }} dpr={[1, 2]} shadows={false}>
        <hemisphereLight args={["#dbe8ff", "#13203a", 1.25]} />
        <directionalLight position={[6, 8, 5]} intensity={1.5} />
        <directionalLight position={[-6, 3, -5]} intensity={0.5} color="#7fb2ff" />
        <Floor />
        <CarBody xray={xray} />
        <Internals statusOf={statusOf} selected={selected} onSelect={onSelect} show={xray} />
        {COMP_ORDER.map((id) => {
          const c = byId(id);
          return (
            <Hotspot
              key={id} id={id} status={statusOf(id)} selected={selected === id} onSelect={onSelect}
              hideDot={xray}
              showLabel={labels === "all" || selected === id}
              detail={c ? `${STATUS_LABEL[c.status]} · SHAP ${signed(c.ai_contribution.value)}` : "…"}
            />
          );
        })}
        <OrbitControls ref={controls} makeDefault enablePan={false} minDistance={2.6} maxDistance={11}
                       maxPolarAngle={Math.PI / 2.05} target={DEFAULT_VIEW.target} autoRotate={autoRotate} autoRotateSpeed={1.2} />
        <Rig focus={focus} controls={controls} />
      </Canvas>
      {legend && (
        <div className="legend">
          {["critical", "watch", "ok"].map((s) => (
            <span key={s}><i style={{ background: STATUS_COLOR[s] }} />{STATUS_LABEL[s]}</span>
          ))}
        </div>
      )}
    </div>
  );
}
