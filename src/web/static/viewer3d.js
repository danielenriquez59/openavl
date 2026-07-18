/**
 * Three.js 3D aircraft viewer with optional Cp pressure overlay via vertex colors.
 */

import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

/** Diverging blue → white → red colormap for ΔCp. */
function dcpToColor(value, vmin, vmax) {
  const t = vmax > vmin ? (value - vmin) / (vmax - vmin) : 0.5;
  const clamped = Math.max(0, Math.min(1, t));
  let r;
  let g;
  let b;
  if (clamped < 0.5) {
    const u = clamped * 2;
    r = u;
    g = u;
    b = 1;
  } else {
    const u = (clamped - 0.5) * 2;
    r = 1;
    g = 1 - u;
    b = 1 - u;
  }
  return new THREE.Color(r, g, b);
}

/** Build a filled-circle texture for the CG sprite marker. */
function createCircleTexture({ fill, stroke, strokeWidth = 4, size = 28 }) {
  const canvas = document.createElement("canvas");
  canvas.width = size;
  canvas.height = size;
  const ctx = canvas.getContext("2d");
  if (!ctx) return null;

  const center = size / 2;
  const radius = center - 3;
  ctx.beginPath();
  ctx.arc(center, center, radius, 0, Math.PI * 2);
  ctx.fillStyle = fill;
  ctx.fill();
  ctx.lineWidth = strokeWidth;
  ctx.strokeStyle = stroke;
  ctx.stroke();

  const texture = new THREE.CanvasTexture(canvas);
  texture.needsUpdate = true;
  return texture;
}

/** Build a filled-circle texture for the CG sprite marker. */
function createCgCircleTexture() {
  return createCircleTexture({ fill: "#f2c94c", stroke: "#b8860b" });
}

/** Build a filled-circle texture for individual mass component markers. */
function createMassCircleTexture() {
  return createCircleTexture({ fill: "#3b82f6", stroke: "#bfdbfe", strokeWidth: 3, size: 22 });
}

/**
 * Swap horizontal/vertical pointer axes for OrbitControls rotate drags.
 *
 * AVL uses Z-up; default OrbitControls maps mouse X to azimuth (around Z) and
 * mouse Y to elevation, which feels like pitch/yaw are swapped in the viewer.
 * Relaying rotate pointer events with X/Y swapped restores natural orbit.
 *
 * @param {HTMLElement} domElement - Canvas element used by OrbitControls.
 */
function installZUpOrbitAxisSwap(domElement) {
  let rotateDrag = false;

  const isMouseRotateDown = (event) =>
    event.pointerType === "mouse"
    && event.button === 0
    && !(event.ctrlKey || event.metaKey || event.shiftKey);

  const swapPointerEvent = (event) => {
    const swapped = new PointerEvent(event.type, {
      bubbles: true,
      cancelable: event.cancelable,
      composed: event.composed,
      pointerId: event.pointerId,
      pointerType: event.pointerType,
      isPrimary: event.isPrimary,
      width: event.width,
      height: event.height,
      pressure: event.pressure,
      button: event.button,
      buttons: event.buttons,
      clientX: event.clientY,
      clientY: event.clientX,
      screenX: event.screenY,
      screenY: event.screenX,
      pageX: event.pageY,
      pageY: event.pageX,
      movementX: event.movementY,
      movementY: event.movementX,
      ctrlKey: event.ctrlKey,
      shiftKey: event.shiftKey,
      altKey: event.altKey,
      metaKey: event.metaKey,
    });
    swapped.__orbitAxisSwapped = true;
    return swapped;
  };

  const relaySwappedPointer = (event) => {
    event.stopImmediatePropagation();
    domElement.dispatchEvent(swapPointerEvent(event));
  };

  domElement.addEventListener("pointerdown", (event) => {
    if (event.__orbitAxisSwapped) return;
    if (isMouseRotateDown(event)) rotateDrag = true;
    if (rotateDrag) relaySwappedPointer(event);
  }, true);

  domElement.addEventListener("pointermove", (event) => {
    if (event.__orbitAxisSwapped) return;
    if (rotateDrag) relaySwappedPointer(event);
  }, true);

  const endRotateDrag = (event) => {
    if (event.__orbitAxisSwapped) return;
    if (rotateDrag) relaySwappedPointer(event);
    rotateDrag = false;
  };

  domElement.addEventListener("pointerup", endRotateDrag, true);
  domElement.addEventListener("pointercancel", () => {
    rotateDrag = false;
  }, true);
}

/**
 * WebGL aircraft geometry viewer backed by Three.js.
 */
export class AircraftViewer3D {
  /**
   * @param {HTMLElement} container - DOM element that hosts the canvas.
   */
  constructor(container) {
    this.container = container;
    this.meshes = [];
    this.surfaceMeshes = new Map();
    this.showLift = false;
    this.showWake = false;
    this.wireframeOnly = false;
    this.showCg = false;
    this.liftData = null;
    this.wakeData = null;
    this.cgPoint = null;
    this.componentMasses = [];
    this.liftGroup = null;
    this.wakeGroup = null;
    this.cgGroup = null;

    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0x0a0d12);

    const aspect = Math.max(container.clientWidth / Math.max(container.clientHeight, 1), 1);
    this.camera = new THREE.PerspectiveCamera(45, aspect, 0.01, 5000);
    this.camera.position.set(8, 4, 10);

    this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
    this.renderer.setPixelRatio(window.devicePixelRatio);
    this.renderer.setSize(container.clientWidth, container.clientHeight);
    container.appendChild(this.renderer.domElement);

    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.08;
    installZUpOrbitAxisSwap(this.renderer.domElement);

    const ambient = new THREE.AmbientLight(0xffffff, 0.55);
    const key = new THREE.DirectionalLight(0xffffff, 0.85);
    key.position.set(5, 10, 7);
    const fill = new THREE.DirectionalLight(0x7fb3ff, 0.25);
    fill.position.set(-6, 2, -4);
    this.scene.add(ambient, key, fill);

    const axes = new THREE.AxesHelper(2);
    this.scene.add(axes);

    this._boundResize = () => this._onResize();
    window.addEventListener("resize", this._boundResize);

    this._createOverlay();

    this._animate();
  }

  /** Create toggle buttons overlaid on the 3D viewer canvas. */
  _createOverlay() {
    this.overlay = document.createElement("div");
    this.overlay.className = "viewer-overlay";

    this.btnLift = document.createElement("button");
    this.btnLift.type = "button";
    this.btnLift.textContent = "Lift";
    this.btnLift.title = "Toggle spanwise lift distribution";
    this.btnLift.addEventListener("click", () => this.setShowLift(!this.showLift));

    this.btnWake = document.createElement("button");
    this.btnWake.type = "button";
    this.btnWake.textContent = "Wake";
    this.btnWake.title = "Toggle trailing wake filaments";
    this.btnWake.addEventListener("click", () => this.setShowWake(!this.showWake));

    this.btnMesh = document.createElement("button");
    this.btnMesh.type = "button";
    this.btnMesh.textContent = "Mesh";
    this.btnMesh.title = "Toggle aerodynamic panel mesh";
    this.btnMesh.addEventListener("click", () => this.setWireframeOnly(!this.wireframeOnly));

    this.btnCg = document.createElement("button");
    this.btnCg.type = "button";
    this.btnCg.dataset.overlay = "cg";
    this.btnCg.textContent = "CG";
    this.btnCg.title = "Toggle center-of-gravity marker";
    this.btnCg.addEventListener("click", () => this.setShowCg(!this.showCg));

    this.overlay.append(this.btnLift, this.btnWake, this.btnMesh, this.btnCg);
    this.container.appendChild(this.overlay);

    this.viewOverlay = document.createElement("div");
    this.viewOverlay.className = "viewer-overlay viewer-overlay-views";
    for (const axis of ["x", "y", "z"]) {
      const btn = document.createElement("button");
      btn.type = "button";
      btn.dataset.overlay = "view";
      btn.textContent = `+${axis.toUpperCase()}`;
      btn.title = `View from +${axis.toUpperCase()} axis`;
      btn.addEventListener("click", () => this.setViewFromAxis(axis));
      this.viewOverlay.appendChild(btn);
    }
    this.container.appendChild(this.viewOverlay);
  }

  /**
   * Show or hide the spanwise lift distribution overlay.
   *
   * @param {boolean} show
   */
  setShowLift(show) {
    this.showLift = show;
    this.btnLift?.classList.toggle("active", show);
    this._rebuildLiftOverlay();
  }

  /**
   * Show or hide the trailing wake filament overlay.
   *
   * @param {boolean} show
   */
  setShowWake(show) {
    this.showWake = show;
    this.btnWake?.classList.toggle("active", show);
    this._rebuildWakeOverlay();
  }

  /**
   * Toggle between filled geometry and wireframe-only rendering.
   *
   * @param {boolean} enabled
   */
  setWireframeOnly(enabled) {
    this.wireframeOnly = enabled;
    this.btnMesh?.classList.toggle("active", enabled);
    for (const object of this.meshes) {
      if (object.isMesh) {
        object.visible = !enabled;
      } else if (object.userData.aerodynamicPanelMesh) {
        object.visible = enabled;
      } else if (object.isLineSegments) {
        object.visible = !enabled;
      }
    }
  }

  /**
   * Show or hide the center-of-gravity marker.
   *
   * @param {boolean} show
   */
  setShowCg(show) {
    this.showCg = show;
    this.btnCg?.classList.toggle("active", show);
    this._rebuildCgOverlay();
  }

  /**
   * Update strip loading data from a solve result.
   *
   * @param {{
   *   surfaces?: Array<{
   *     name?: string,
   *     strips?: Array<{
   *       ensy?: number,
   *       ensz?: number,
   *       points?: Array<{ x: number, y: number, z: number, cl?: number }>
   *     }>
   *   }>,
   *   cref?: number,
   *   bref?: number
   * }|null} data
   */
  updateLiftDistribution(data) {
    this.liftData = data;
    this._rebuildLiftOverlay();
  }

  /**
   * Update trailing wake filament data from a solve result.
   *
   * @param {{
   *   surfaces?: Array<{
   *     name?: string,
   *     filaments?: Array<{
   *       x0?: number, y0?: number, z0?: number,
   *       x1?: number, y1?: number, z1?: number
   *     }>
   *   }>
   * }|null} data
   */
  updateWake(data) {
    this.wakeData = data;
    this._rebuildWakeOverlay();
  }

  /**
   * Update the center-of-gravity marker position.
   *
   * @param {{ x?: number, y?: number, z?: number }|null} cg
   */
  updateCg(cg) {
    if (cg && Number.isFinite(cg.x) && Number.isFinite(cg.y) && Number.isFinite(cg.z)) {
      this.cgPoint = { x: cg.x, y: cg.y, z: cg.z };
    } else {
      this.cgPoint = null;
    }
    this._rebuildCgOverlay();
  }

  /**
   * Update component mass marker positions from the loaded mass file.
   *
   * @param {Array<{ mass?: number, x?: number, y?: number, z?: number }>} masses
   */
  updateComponentMasses(masses) {
    this.componentMasses = Array.isArray(masses)
      ? masses
          .map((mass) => ({
            mass: Number(mass.mass),
            x: Number(mass.x),
            y: Number(mass.y),
            z: Number(mass.z),
          }))
          .filter((mass) => Number.isFinite(mass.x) && Number.isFinite(mass.y) && Number.isFinite(mass.z))
      : [];
    this._rebuildCgOverlay();
  }

  /** Remove a scene overlay group and dispose its GPU resources. */
  _disposeOverlayGroup(group) {
    if (!group) return;
    this.scene.remove(group);
    group.traverse((obj) => {
      obj.geometry?.dispose?.();
      if (Array.isArray(obj.material)) {
        obj.material.forEach((mat) => mat.dispose?.());
      } else {
        obj.material?.dispose?.();
      }
    });
  }

  /** Rebuild lift-distribution vector and spanwise line overlays. */
  _rebuildLiftOverlay() {
    this._disposeOverlayGroup(this.liftGroup);
    this.liftGroup = null;
    if (!this.showLift || !this.liftData?.surfaces?.length) return;

    const cref = Number(this.liftData.cref) || 1;
    const bref = Number(this.liftData.bref) || 1;
    const scale = Math.min(0.55 * cref, 0.14 * bref);

    const vectorVerts = [];
    const spanVerts = [];

    for (const surf of this.liftData.surfaces) {
      for (const strip of surf.strips ?? []) {
        const ensy = Number(strip.ensy) || 0;
        const ensz = Number(strip.ensz) || 0;
        let prevLoad = null;

        for (const pt of strip.points ?? []) {
          const x = Number(pt.x);
          const y = Number(pt.y);
          const z = Number(pt.z);
          const cl = Number(pt.cl) || 0;
          if (!Number.isFinite(x) || !Number.isFinite(y) || !Number.isFinite(z)) continue;

          const delta = cl * scale;
          const xLoad = x;
          const yLoad = y + delta * ensy;
          const zLoad = z + delta * ensz;

          vectorVerts.push(x, y, z, xLoad, yLoad, zLoad);
          if (prevLoad) {
            spanVerts.push(...prevLoad, xLoad, yLoad, zLoad);
          }
          prevLoad = [xLoad, yLoad, zLoad];
        }
      }
    }

    if (!vectorVerts.length && !spanVerts.length) return;

    this.liftGroup = new THREE.Group();
    this.liftGroup.name = "lift-distribution";
    this.liftGroup.renderOrder = 2;

    if (vectorVerts.length) {
      const vecGeom = new THREE.BufferGeometry();
      vecGeom.setAttribute("position", new THREE.Float32BufferAttribute(vectorVerts, 3));
      const vecLines = new THREE.LineSegments(
        vecGeom,
        new THREE.LineBasicMaterial({ color: 0x22c55e, transparent: true, opacity: 0.9 }),
      );
      vecLines.name = "lift-vectors";
      this.liftGroup.add(vecLines);
    }

    if (spanVerts.length) {
      const spanGeom = new THREE.BufferGeometry();
      spanGeom.setAttribute("position", new THREE.Float32BufferAttribute(spanVerts, 3));
      const spanLines = new THREE.LineSegments(
        spanGeom,
        new THREE.LineBasicMaterial({ color: 0xef4444, transparent: true, opacity: 0.9 }),
      );
      spanLines.name = "lift-span-lines";
      this.liftGroup.add(spanLines);
    }

    this.scene.add(this.liftGroup);
  }

  /** Rebuild the trailing wake filament overlay. */
  _rebuildWakeOverlay() {
    this._disposeOverlayGroup(this.wakeGroup);
    this.wakeGroup = null;
    if (!this.showWake || !this.wakeData?.surfaces?.length) return;

    const vertices = [];
    for (const surface of this.wakeData.surfaces) {
      for (const filament of surface.filaments ?? []) {
        const coordinates = [
          Number(filament.x0),
          Number(filament.y0),
          Number(filament.z0),
          Number(filament.x1),
          Number(filament.y1),
          Number(filament.z1),
        ];
        if (coordinates.every(Number.isFinite)) {
          vertices.push(...coordinates);
        }
      }
    }

    if (!vertices.length) return;

    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute("position", new THREE.Float32BufferAttribute(vertices, 3));
    const lines = new THREE.LineSegments(
      geometry,
      new THREE.LineBasicMaterial({ color: 0x38bdf8, transparent: true, opacity: 0.72 }),
    );
    lines.name = "wake-filaments";

    this.wakeGroup = new THREE.Group();
    this.wakeGroup.name = "trailing-wake";
    this.wakeGroup.renderOrder = 2;
    this.wakeGroup.add(lines);
    this.scene.add(this.wakeGroup);
  }

  /** Rebuild the center-of-gravity marker mesh. */
  _rebuildCgOverlay() {
    this._disposeOverlayGroup(this.cgGroup);
    this.cgGroup = null;
    if (!this.showCg || !this.cgPoint) return;

    const { x, y, z } = this.cgPoint;
    this.cgGroup = new THREE.Group();
    this.cgGroup.name = "cg-marker";
    this.cgGroup.renderOrder = 3;

    const box = this._getModelBounds();
    const size = box ? box.getSize(new THREE.Vector3()) : new THREE.Vector3(1, 1, 1);
    const maxDim = Math.max(size.x, size.y, size.z, 0.1);
    const markerSize = maxDim * 0.045;
    const massMarkerSize = markerSize * 0.55;

    if (!this._cgSpriteTexture) {
      this._cgSpriteTexture = createCgCircleTexture();
    }

    const sprite = new THREE.Sprite(
      new THREE.SpriteMaterial({
        map: this._cgSpriteTexture ?? undefined,
        color: this._cgSpriteTexture ? 0xffffff : 0xf2c94c,
        transparent: true,
        opacity: 0.98,
        depthTest: true,
        depthWrite: false,
      }),
    );
    sprite.position.set(x, y, z);
    sprite.scale.set(markerSize, markerSize, 1);
    sprite.renderOrder = 3;
    this.cgGroup.add(sprite);

    if (!this._massSpriteTexture) {
      this._massSpriteTexture = createMassCircleTexture();
    }

    for (const mass of this.componentMasses) {
      const massSprite = new THREE.Sprite(
        new THREE.SpriteMaterial({
          map: this._massSpriteTexture ?? undefined,
          color: this._massSpriteTexture ? 0xffffff : 0x3b82f6,
          transparent: true,
          opacity: 0.96,
          depthTest: true,
          depthWrite: false,
        }),
      );
      massSprite.position.set(mass.x, mass.y, mass.z);
      massSprite.scale.set(massMarkerSize, massMarkerSize, 1);
      massSprite.renderOrder = 4;
      massSprite.name = "component-mass-marker";
      if (Number.isFinite(mass.mass)) {
        massSprite.userData.mass = mass.mass;
      }
      this.cgGroup.add(massSprite);
    }

    this.scene.add(this.cgGroup);
  }

  /** Render loop. */
  _animate() {
    requestAnimationFrame(() => this._animate());
    this.controls.update();
    this.renderer.render(this.scene, this.camera);
  }

  /**
   * Return the axis-aligned bounding box of loaded mesh geometry.
   *
   * @returns {THREE.Box3|null}
   */
  _getModelBounds() {
    const box = new THREE.Box3();
    let hasGeometry = false;
    for (const mesh of this.meshes) {
      if (!mesh.isMesh) continue;
      box.expandByObject(mesh);
      hasGeometry = true;
    }
    return hasGeometry ? box : null;
  }

  /**
   * Snap the camera to a standard axis-aligned view of the model.
   *
   * @param {"x"|"y"|"z"} axis - View direction from the positive axis toward the model.
   */
  setViewFromAxis(axis) {
    const box = this._getModelBounds();
    if (!box) return;

    const center = box.getCenter(new THREE.Vector3());
    const size = box.getSize(new THREE.Vector3());
    const maxDim = Math.max(size.x, size.y, size.z, 0.1);
    const dist = maxDim * 1.8;

    const axisDirs = {
      x: new THREE.Vector3(1, 0, 0),
      y: new THREE.Vector3(0, 1, 0),
      z: new THREE.Vector3(0, 0, 1),
    };
    const dir = axisDirs[axis];
    if (!dir) return;

    const upVectors = {
      x: new THREE.Vector3(0, 0, 1),
      y: new THREE.Vector3(0, 0, 1),
      z: new THREE.Vector3(0, 1, 0),
    };

    this.controls.target.copy(center);
    this.camera.up.copy(upVectors[axis]);
    this.camera.position.copy(center).add(dir.multiplyScalar(dist));
    this.camera.lookAt(center);
    this.camera.near = maxDim * 0.01;
    this.camera.far = maxDim * 50;
    this.camera.updateProjectionMatrix();
    this.controls.update();
  }

  /** Fit camera to current mesh bounds. */
  fitToModel() {
    const box = this._getModelBounds();
    if (!box) return;

    const center = box.getCenter(new THREE.Vector3());
    const size = box.getSize(new THREE.Vector3());
    const maxDim = Math.max(size.x, size.y, size.z, 0.1);
    const dist = maxDim * 1.8;

    this.controls.target.copy(center);
    this.camera.up.set(0, 0, 1);
    this.camera.position.set(center.x + dist * 0.6, center.y + dist * 0.35, center.z + dist * 0.7);
    this.camera.near = maxDim * 0.01;
    this.camera.far = maxDim * 50;
    this.camera.updateProjectionMatrix();
    this.controls.update();
  }

  /**
   * Load geometry exported from the server.
   *
   * @param {{
   *   surfaces: Array<{
   *     name: string,
   *     color?: number[],
   *     positions: number[],
   *     indices: number[],
   *     panel_lines?: number[],
   *     dcp?: number[]
   *   }>,
   *   bodies?: Array<{
   *     name: string,
   *     color?: number[],
   *     positions: number[],
   *     indices: number[]
   *   }>
   * }} geometry
   */
  loadGeometry(geometry) {
    this._clearMeshes();

    const surfaces = geometry?.surfaces ?? [];
    for (const surf of surfaces) {
      this._addMesh(surf, { isBody: false });
    }

    const bodies = geometry?.bodies ?? [];
    for (const body of bodies) {
      this._addMesh(body, { isBody: true });
    }

    this.setWireframeOnly(this.wireframeOnly);
    this.fitToModel();
  }

  /**
   * Add one surface or body mesh to the scene.
   *
   * @param {{ name?: string, color?: number[], positions: number[], indices: number[], panel_lines?: number[], dcp?: number[] }} meshData
   * @param {{ isBody?: boolean }} options
   */
  _addMesh(meshData, { isBody = false } = {}) {
    const geom = new THREE.BufferGeometry();
    const positions = new Float32Array(meshData.positions);
    if (positions.length < 9) return;

    geom.setAttribute("position", new THREE.BufferAttribute(positions, 3));
    geom.setIndex(meshData.indices);

    const vertexCount = positions.length / 3;
    const defaultColor = isBody ? new THREE.Color(0x6c757d) : new THREE.Color(0x1f77b4);
    const baseColor = meshData.color?.length >= 3
      ? new THREE.Color(meshData.color[0], meshData.color[1], meshData.color[2])
      : defaultColor;

    const colors = new Float32Array(vertexCount * 3);
    for (let i = 0; i < vertexCount; i += 1) {
      colors[i * 3] = baseColor.r;
      colors[i * 3 + 1] = baseColor.g;
      colors[i * 3 + 2] = baseColor.b;
    }
    geom.setAttribute("color", new THREE.BufferAttribute(colors, 3));

    if (!isBody && meshData.dcp?.length) {
      this._applyDcpToGeometry(geom, meshData.dcp);
    }

    geom.computeVertexNormals();

    const material = new THREE.MeshPhongMaterial({
      vertexColors: true,
      side: THREE.DoubleSide,
      shininess: isBody ? 8 : 20,
      flatShading: false,
      transparent: isBody,
      opacity: isBody ? 0.88 : 1.0,
    });

    const mesh = new THREE.Mesh(geom, material);
    mesh.name = meshData.name ?? (isBody ? "body" : "surface");
    this.scene.add(mesh);
    this.meshes.push(mesh);
    if (!isBody) {
      this.surfaceMeshes.set(meshData.name ?? `surface_${this.meshes.length}`, mesh);
    }

    const edges = new THREE.EdgesGeometry(geom, 15);
    const line = new THREE.LineSegments(
      edges,
      new THREE.LineBasicMaterial({
        color: isBody ? 0x3a4048 : 0x2a313c,
        transparent: true,
        opacity: isBody ? 0.25 : 0.35,
      }),
    );
    this.scene.add(line);
    this.meshes.push(line);

    if (!isBody && meshData.panel_lines?.length >= 6) {
      const panelGeometry = new THREE.BufferGeometry();
      panelGeometry.setAttribute(
        "position",
        new THREE.Float32BufferAttribute(meshData.panel_lines, 3),
      );
      const panelLines = new THREE.LineSegments(
        panelGeometry,
        new THREE.LineBasicMaterial({ color: 0xa9bed3, transparent: true, opacity: 0.95 }),
      );
      panelLines.name = `${mesh.name}-aerodynamic-panels`;
      panelLines.userData.aerodynamicPanelMesh = true;
      panelLines.visible = this.wireframeOnly;
      this.scene.add(panelLines);
      this.meshes.push(panelLines);
    }
  }

  /**
   * Update Cp overlay from per-surface or flat dcp arrays.
   *
   * @param {Array<{ name?: string, dcp: number[] }>|number[]} cpData
   */
  updateCpOverlay(cpData) {
    if (!cpData) return;

    if (Array.isArray(cpData) && cpData.length && typeof cpData[0] === "object" && cpData[0].dcp) {
      for (const entry of cpData) {
        const mesh = entry.name ? this.surfaceMeshes.get(entry.name) : null;
        if (mesh?.geometry && entry.dcp) {
          this._applyDcpToGeometry(mesh.geometry, entry.dcp);
        }
      }
      return;
    }

    const meshList = [...this.surfaceMeshes.values()].filter((m) => m.isMesh);
    if (meshList.length === 1 && Array.isArray(cpData)) {
      this._applyDcpToGeometry(meshList[0].geometry, cpData);
    }
  }

  /**
   * Map dcp values to vertex colors on a BufferGeometry.
   *
   * @param {THREE.BufferGeometry} geometry
   * @param {number[]} dcp
   */
  _applyDcpToGeometry(geometry, dcp) {
    const posAttr = geometry.getAttribute("position");
    if (!posAttr) return;

    const vertexCount = posAttr.count;
    let vmin = Infinity;
    let vmax = -Infinity;
    for (let i = 0; i < Math.min(vertexCount, dcp.length); i += 1) {
      const v = dcp[i];
      if (Number.isFinite(v)) {
        vmin = Math.min(vmin, v);
        vmax = Math.max(vmax, v);
      }
    }
    if (!Number.isFinite(vmin)) {
      vmin = -0.5;
      vmax = 0.5;
    }

    const colors = new Float32Array(vertexCount * 3);
    for (let i = 0; i < vertexCount; i += 1) {
      const c = dcp[i] ?? 0;
      const color = dcpToColor(c, vmin, vmax);
      colors[i * 3] = color.r;
      colors[i * 3 + 1] = color.g;
      colors[i * 3 + 2] = color.b;
    }
    geometry.setAttribute("color", new THREE.BufferAttribute(colors, 3));
    geometry.attributes.color.needsUpdate = true;
  }

  /** Remove all surface meshes from the scene. */
  _clearMeshes() {
    for (const obj of this.meshes) {
      this.scene.remove(obj);
      obj.geometry?.dispose?.();
      obj.material?.dispose?.();
    }
    this.meshes = [];
    this.surfaceMeshes.clear();
    this._rebuildCgOverlay();
  }

  /** Resize renderer when the container changes size. */
  _onResize() {
    const w = this.container.clientWidth;
    const h = this.container.clientHeight;
    if (w < 1 || h < 1) return;
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(w, h);
  }

  /** Release WebGL resources. */
  dispose() {
    window.removeEventListener("resize", this._boundResize);
    this._disposeOverlayGroup(this.liftGroup);
    this._disposeOverlayGroup(this.wakeGroup);
    this._disposeOverlayGroup(this.cgGroup);
    this.liftGroup = null;
    this.wakeGroup = null;
    this.cgGroup = null;
    this._cgSpriteTexture?.dispose?.();
    this._cgSpriteTexture = null;
    this._massSpriteTexture?.dispose?.();
    this._massSpriteTexture = null;
    this.overlay?.remove();
    this.viewOverlay?.remove();
    this._clearMeshes();
    this.renderer.dispose();
    this.container.removeChild(this.renderer.domElement);
  }
}
