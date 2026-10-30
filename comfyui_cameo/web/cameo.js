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
