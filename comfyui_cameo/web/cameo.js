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
  ["Apollo 11: planting the flag", "Photobomb the moon landing"],
  ["The Beatles: rooftop concert", "Air guitar next to the band"],
  ["Saturday Night Fever: Tony's solo", "Challenge him to a dance-off"],
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
  const making = cast?.widgets?.find((w) => w.name === "make_cameo")?.value;
  if (!making) app.queuePrompt(0, 1); // a preview only: about 15 seconds, no models loaded
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
      <li><b>2 · Check the preview</b>The clip and its title show up here in about 15 seconds.</li>
      <li><b>3 · Make your cameo</b>Add your photo (or webcam), switch on <i>make my cameo</i>, and Run.</li>
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
