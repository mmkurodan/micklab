"use strict";

(function registerBrowserApiWorker() {
  if (!("serviceWorker" in navigator)) {
    return;
  }

  const ensureRegistration = async () => {
    const registration = await navigator.serviceWorker.register("/browser-api-sw.js", {
      scope: "/",
    });
    if (registration.waiting) {
      registration.waiting.postMessage({ type: "SKIP_WAITING" });
    }
    return navigator.serviceWorker.ready;
  };

  if (document.readyState === "complete") {
    ensureRegistration().catch(() => {});
    return;
  }
  window.addEventListener("load", () => {
    ensureRegistration().catch(() => {});
  });
})();
