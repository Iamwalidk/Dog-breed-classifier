function formatFileSize(bytes) {
  if (!Number.isFinite(bytes) || bytes < 0) {
    return "Unknown size";
  }

  const units = ["B", "KB", "MB", "GB"];
  let value = bytes;
  let index = 0;
  while (value >= 1024 && index < units.length - 1) {
    value /= 1024;
    index += 1;
  }
  return `${value.toFixed(value >= 10 || index === 0 ? 0 : 1)} ${units[index]}`;
}

function getFileLabel(file) {
  if (!file) {
    return "No image selected yet.";
  }

  const name = file.name || "Selected file";
  const size = formatFileSize(file.size);
  return `${name} - ${size}`;
}

function isImageFile(file) {
  if (!file) {
    return false;
  }

  if (typeof file.type === "string" && file.type.startsWith("image/")) {
    return true;
  }

  const name = file.name || "";
  return /\.(png|jpe?g|webp|bmp)$/i.test(name);
}

function setInputFile(input, file) {
  if (!input) {
    return;
  }

  try {
    const dataTransfer = new DataTransfer();
    dataTransfer.items.add(file);
    input.files = dataTransfer.files;
  } catch {
    // Older browsers may not allow programmatic assignment. The UI still works
    // because the user can select a file directly from the picker.
  }
}

function setDropzoneState(dropzone, isActive) {
  if (!dropzone) {
    return;
  }
  dropzone.classList.toggle("is-dragover", Boolean(isActive));
}

export function createUploadCardController({
  fileInput,
  previewImage,
  previewWrapper,
  placeholder,
  analyzeButton,
  resetButton,
  fileMeta,
  onSelectionChange,
  onError,
}) {
  let previewUrl = null;
  let currentFile = null;
  let dragDepth = 0;

  function emitChange(file) {
    if (typeof onSelectionChange === "function") {
      onSelectionChange(file);
    }
  }

  function emitError(message) {
    if (typeof onError === "function" && message) {
      onError(message);
    }
  }

  function revokePreviewUrl() {
    if (previewUrl) {
      URL.revokeObjectURL(previewUrl);
      previewUrl = null;
    }
  }

  function updateMeta(file) {
    if (fileMeta) {
      fileMeta.textContent = getFileLabel(file);
    }
  }

  function updatePreview(file) {
    currentFile = file || null;
    revokePreviewUrl();

    if (!currentFile) {
      previewImage.hidden = true;
      previewImage.removeAttribute("src");
      placeholder.hidden = false;
      previewWrapper.classList.remove("has-image");
      analyzeButton.disabled = true;
      resetButton.hidden = true;
      updateMeta(null);
      emitChange(null);
      return;
    }

    previewUrl = URL.createObjectURL(currentFile);
    previewImage.src = previewUrl;
    previewImage.hidden = false;
    placeholder.hidden = true;
    previewWrapper.classList.add("has-image");
    analyzeButton.disabled = false;
    resetButton.hidden = false;
    updateMeta(currentFile);
    emitChange(currentFile);
  }

  function clearSelection() {
    currentFile = null;
    fileInput.value = "";
    updatePreview(null);
  }

  function setLoading(isLoading) {
    fileInput.disabled = isLoading;
    analyzeButton.disabled = isLoading || !currentFile;
    resetButton.disabled = isLoading;
    analyzeButton.textContent = isLoading ? "Analyzing..." : "Analyze image";
    previewWrapper.setAttribute("aria-busy", String(Boolean(isLoading)));
  }

  function handleSelection(file) {
    if (!file) {
      clearSelection();
      return true;
    }

    if (!isImageFile(file)) {
      clearSelection();
      emitError("Please choose an image file such as PNG, JPG, JPEG, WEBP, or BMP.");
      return false;
    }

    updatePreview(file);
    return true;
  }

  function handleDrop(event) {
    event.preventDefault();
    event.stopPropagation();
    dragDepth = 0;
    setDropzoneState(previewWrapper, false);

    const file = event.dataTransfer && event.dataTransfer.files ? event.dataTransfer.files[0] : null;
    if (!file) {
      return;
    }

    setInputFile(fileInput, file);
    handleSelection(file);
  }

  fileInput.addEventListener("change", () => {
    const file = fileInput.files && fileInput.files[0] ? fileInput.files[0] : null;
    if (file) {
      handleSelection(file);
      return;
    }
    clearSelection();
  });

  fileInput.addEventListener("dragenter", (event) => {
    event.preventDefault();
  });

  previewWrapper.addEventListener("dragenter", (event) => {
    event.preventDefault();
    dragDepth += 1;
    setDropzoneState(previewWrapper, true);
  });

  previewWrapper.addEventListener("dragover", (event) => {
    event.preventDefault();
    setDropzoneState(previewWrapper, true);
  });

  previewWrapper.addEventListener("dragleave", (event) => {
    event.preventDefault();
    dragDepth = Math.max(0, dragDepth - 1);
    if (dragDepth === 0) {
      setDropzoneState(previewWrapper, false);
    }
  });

  previewWrapper.addEventListener("drop", handleDrop);

  resetButton.addEventListener("click", clearSelection);

  updatePreview(null);

  return {
    clearSelection,
    getSelectedFile: () => currentFile,
    setLoading,
    handleSelection,
  };
}
