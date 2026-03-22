function createLoaderElement(message) {
  const wrapper = document.createElement("div");
  wrapper.className = "ai-loader";
  wrapper.setAttribute("role", "status");
  wrapper.setAttribute("aria-live", "polite");

  const head = document.createElement("div");
  head.className = "ai-loader-head";

  const spinner = document.createElement("div");
  spinner.className = "ai-loader-spinner";
  spinner.setAttribute("aria-hidden", "true");

  const textWrap = document.createElement("div");
  textWrap.className = "ai-loader-copy";

  const title = document.createElement("p");
  title.className = "ai-loader-title";
  title.textContent = "Analyzing breed characteristics...";

  const subtitle = document.createElement("p");
  subtitle.className = "ai-loader-subtitle";
  subtitle.textContent = message || "Running model inference, ranking predictions, and preparing the result cards.";

  const dots = document.createElement("span");
  dots.className = "ai-loader-dots";
  dots.setAttribute("aria-hidden", "true");
  dots.innerHTML = "<i></i><i></i><i></i>";

  textWrap.append(title, subtitle, dots);
  head.append(spinner, textWrap);

  const bars = document.createElement("div");
  bars.className = "loader-bars";
  ["medium", "short", "tiny"].forEach((size) => {
    const bar = document.createElement("div");
    bar.className = `loader-bar ${size}`;
    bars.append(bar);
  });

  wrapper.append(head, bars);
  return wrapper;
}

export function showLoader(container, message) {
  container.replaceChildren(createLoaderElement(message));
  container.hidden = false;
}

export function hideLoader(container) {
  container.hidden = true;
  container.replaceChildren();
}
