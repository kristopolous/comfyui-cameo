// Cameo: friendlier webcam message when the page isn't a secure context (plain http on a non-localhost address).
import { app } from "../../scripts/app.js";

const MESSAGE =
  "📷 The webcam only works over HTTPS or on localhost.\n\n" +
  "Upload a photo instead (that's selected for you), or open this app at http://localhost:8188 " +
  "or its https:// address.";

function friendly(node) {
  const widget = node.widgets?.find((w) => w.type === "WEBCAM" || w.name === "image");
  const el = widget?.element;
  if (!el) return;
  const box = document.createElement("div");
  box.style.cssText = "padding:12px;color:#ddd;white-space:pre-wrap;font-size:13px;line-height:1.4;text-align:left";
  box.textContent = MESSAGE;
  el.replaceChildren(box);
}

function preferPhoto() {
  for (const node of app.graph?._nodes ?? []) {
    if (node.comfyClass !== "CameoPhoto" && node.type !== "CameoPhoto") continue;
    const source = node.widgets?.find((w) => w.name === "source");
    if (source && source.value !== "photo") {
      source.value = "photo";
      node.setDirtyCanvas?.(true, true);
    }
  }
}

// Insecure page: drop the webcam (and the photo/webcam switch) from App Mode's inputs before the
// workflow loads, so the app shows a plain photo upload instead of a browser error.
function hideWebcamInApp(graphData) {
  const nodes = graphData?.nodes ?? [];
  const webcam = new Set(nodes.filter((n) => n.type === "WebcamCapture").map((n) => String(n.id)));
  const photo = nodes.filter((n) => n.type === "CameoPhoto");
  for (const n of photo) if (Array.isArray(n.widgets_values)) n.widgets_values[0] = "photo";
  const switches = new Set(photo.map((n) => String(n.id)));
  const lin = graphData?.extra?.linearData;
  if (!lin?.inputs) return;
  lin.inputs = lin.inputs.filter(([id, name]) => !webcam.has(String(id)) && !(switches.has(String(id)) && name === "source"));
}

app.registerExtension({
  name: "Cameo.WebcamHelp",
  beforeConfigureGraph(graphData) {
    if (!window.isSecureContext) hideWebcamInApp(graphData);
  },
  nodeCreated(node) {
    if (window.isSecureContext || node.comfyClass !== "WebcamCapture") return;
    // the core widget writes its error asynchronously; replace it once it has
    setTimeout(() => friendly(node), 600);
    setTimeout(() => friendly(node), 2000);
    setTimeout(preferPhoto, 600);
  },
});

// ---- App Mode welcome: replace the generic empty state with Cameo's own, when the Cameo app is open ----

const SCENES = [
  ["Napoleon Dynamite: the dance", "Out-dance him on stage"],
  ["Pulp Fiction: the twist contest", "Do the twist beside Vincent and Mia"],
  ["Rick Astley: Never Gonna Give You Up", "Be the backup dancer, a beat late"],
  ["Michael Jackson: Billie Jean", "Follow him down the light-up sidewalk"],
  ["The Beatles: rooftop concert", "Air guitar next to the band"],
  ["Abbey Road: crossing with Paul McCartney", "Walk past, distracted on your phone"],
];

function cameoNodes() {
  const nodes = app.graph?._nodes ?? [];
  return { fetch: nodes.find((n) => n.comfyClass === "CameoFetch"), cast: nodes.find((n) => n.comfyClass === "CameoCast") };
}

function setWidget(node, name, value) {
  const w = node?.widgets?.find((w) => w.name === name);
  if (!w) return;
  w.value = value;
  w.callback?.(value, app.canvas, node);
  node.setDirtyCanvas?.(true, true);
}

function pickScene(label) {
  const { fetch, cast } = cameoNodes();
  setWidget(fetch, "suggestion", label);
  setWidget(fetch, "url", "");
  setWidget(cast, "make_cameo", false);
  app.queuePrompt(0, 1); // a preview only: about 15 seconds, no models loaded
}

function welcome() {
  const root = document.createElement("div");
  root.className = "cameo-welcome";
  root.innerHTML = `
    <style>
      .cameo-welcome{max-width:880px;margin:0 auto;padding:28px 24px;color:var(--fg-color,#ddd);text-align:left;pointer-events:auto}
      .cameo-welcome h1{font-size:40px;font-weight:800;margin:0 0 6px;color:#fff}
      .cameo-welcome .pitch{font-size:17px;opacity:.85;margin:0 0 22px}
      .cameo-welcome ol{display:flex;gap:14px;list-style:none;padding:0;margin:0 0 26px}
      .cameo-welcome ol li{flex:1;background:rgba(255,255,255,.05);border:1px solid rgba(255,255,255,.08);border-radius:12px;padding:12px 14px;font-size:14px;line-height:1.4}
      .cameo-welcome ol b{display:block;font-size:12px;letter-spacing:.08em;text-transform:uppercase;opacity:.6;margin-bottom:4px}
      .cameo-welcome h2{font-size:13px;letter-spacing:.08em;text-transform:uppercase;opacity:.6;margin:0 0 10px}
      .cameo-welcome .cards{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}
      .cameo-welcome button.card{all:unset;cursor:pointer;background:linear-gradient(160deg,rgba(255,255,255,.08),rgba(255,255,255,.02));border:1px solid rgba(255,255,255,.1);border-radius:12px;padding:14px;transition:transform .12s,border-color .12s}
      .cameo-welcome button.card:hover{transform:translateY(-2px);border-color:rgba(120,170,255,.7)}
      .cameo-welcome .card .t{font-weight:700;font-size:15px;color:#fff;margin-bottom:4px}
      .cameo-welcome .card .d{font-size:13px;opacity:.75}
      .cameo-welcome .foot{margin-top:20px;font-size:12px;opacity:.55}
      @media (max-width:900px){.cameo-welcome .cards{grid-template-columns:repeat(2,1fr)}.cameo-welcome ol{flex-direction:column}}
    </style>
    <h1>Cameo</h1>
    <p class="pitch">Put yourself in any video. You're <i>added</i> to the scene, next to the people already there; every original pixel stays.</p>
    <ol>
      <li><b>1 · Pick a scene</b>Click one below, or paste a link / type a search on the right.</li>
      <li><b>2 · Check the preview</b>Press Run: the clip and its title show up here in about 15 seconds.</li>
      <li><b>3 · Make your cameo</b>Add your photo (or webcam) on the right, then press <i>Make my cameo</i>.</li>
    </ol>
    <h2>Try one of these</h2>
    <div class="cards"></div>
    <p class="foot">A decision model picks the shots and where you stand · FLUX 3 Klein adds you · LTX-2.5 carries you through · SAM3 keeps everyone else in front.</p>`;
  const cards = root.querySelector(".cards");
  for (const [label, blurb] of SCENES) {
    const b = document.createElement("button");
    b.className = "card";
    b.innerHTML = `<div class="t"></div><div class="d"></div>`;
    b.querySelector(".t").textContent = label;
    b.querySelector(".d").textContent = blurb;
    b.addEventListener("click", () => pickScene(label));
    cards.appendChild(b);
  }
  return root;
}

function decorate() {
  const slot = document.querySelector('[data-testid="linear-welcome"]');
  if (!slot) return;
  const ours = slot.querySelector(":scope > .cameo-welcome");
  if (!cameoNodes().fetch) {
    if (ours) { ours.remove(); for (const c of slot.children) c.style.display = ""; slot.style.maxWidth = ""; }
    return;
  }
  if (ours) return;
  for (const c of slot.children) c.style.display = "none";
  slot.style.maxWidth = "none";
  slot.appendChild(welcome());
}

new MutationObserver(() => decorate()).observe(document.body, { childList: true, subtree: true });

// ---- "Make my cameo" button: Run previews; this button generates. Replaces the make-my-cameo toggle in the app. ----

async function makeCameo(button) {
  const { cast } = cameoNodes();
  const toggle = cast?.widgets?.find((w) => w.name === "make_cameo");
  if (!toggle) return;
  button.disabled = true;
  button.textContent = "Making your cameo… (a few minutes)";
  setWidget(cast, "make_cameo", true);
  try {
    await app.queuePrompt(0, 1); // the prompt is serialized here, with generation on
  } finally {
    setWidget(cast, "make_cameo", false); // back to preview-only for the next Run
    setTimeout(() => { button.disabled = false; button.textContent = "Make my cameo ▶"; }, 4000);
  }
}

function addMakeButton() {
  const panel = document.querySelector('[data-testid="linear-center-panel"]');
  const existing = document.querySelector(".cameo-make");
  const show = panel && cameoNodes().fetch && !panel.querySelector('[data-testid="linear-welcome"]');
  if (!show) { existing?.remove(); return; }
  if (existing && panel.contains(existing)) return;
  const bar = document.createElement("div");
  bar.className = "cameo-make";
  bar.style.cssText = "display:flex;flex-wrap:wrap;justify-content:center;gap:12px;align-items:center;" +
                      "padding:10px 12px 14px;pointer-events:auto";
  const b = document.createElement("button");
  b.textContent = "Make my cameo ▶";
  b.style.cssText = "cursor:pointer;border:none;border-radius:10px;padding:10px 22px;font-size:15px;font-weight:700;" +
                    "color:#0b0c0f;background:linear-gradient(135deg,#9fc2ff,#c9a6ff);box-shadow:0 4px 18px rgba(140,160,255,.35)";
  b.addEventListener("click", () => makeCameo(b));
  const hint = document.createElement("span");
  hint.textContent = "Right video? Add your photo on the right, then";
  hint.style.cssText = "font-size:13px;opacity:.7";
  bar.append(hint, b);
  panel.append(bar); // under the preview, clear of the App Mode toolbar
}

new MutationObserver(() => addMakeButton()).observe(document.body, { childList: true, subtree: true });

// ---- Trim: scrub the preview, then "Start here" / "End here" write exact times into start / end ----

import { api } from "../../scripts/api.js";

let clipStart = 0;
api.addEventListener("executed", (e) => {
  const clip = e.detail?.output?.cameo_clip?.[0];
  if (clip) clipStart = clip.start || 0;
});

function stamp(seconds) {
  const s = Math.max(0, seconds);
  const m = Math.floor(s / 60);
  return `${m}:${(s - m * 60).toFixed(1).padStart(4, "0")}`;
}

function previewVideo() {
  return [...document.querySelectorAll('[data-testid="linear-center-panel"] video')]
    .find((v) => v.getBoundingClientRect().width > 200);
}

function trimButton(label, widget) {
  const b = document.createElement("button");
  b.textContent = label;
  b.style.cssText = "cursor:pointer;border:1px solid rgba(255,255,255,.2);border-radius:8px;padding:7px 12px;" +
                    "font-size:13px;color:#eee;background:rgba(255,255,255,.06)";
  b.addEventListener("click", () => {
    const v = previewVideo();
    const { fetch } = cameoNodes();
    if (!v || !fetch) return;
    const t = stamp(clipStart + v.currentTime);
    setWidget(fetch, widget, t); // an end before the start is ignored by Cameo Fetch
    b.textContent = `${label} ✓ ${t}`;
    setTimeout(() => (b.textContent = label), 2500);
  });
  return b;
}

function addTrimButtons() {
  const bar = document.querySelector(".cameo-make");
  if (!bar || bar.querySelector(".cameo-trim")) return;
  const trim = document.createElement("span");
  trim.className = "cameo-trim";
  trim.style.cssText = "display:inline-flex;gap:8px;margin-right:14px;align-items:center";
  const hint = document.createElement("span");
  hint.textContent = "Trim:";
  hint.style.cssText = "font-size:13px;opacity:.7";
  trim.append(hint, trimButton("Start here", "start"), trimButton("End here", "end"));
  bar.prepend(trim);
}

new MutationObserver(() => addTrimButtons()).observe(document.body, { childList: true, subtree: true });
