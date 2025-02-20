const fileInput = document.getElementById('fileInput');
const predictBtn = document.getElementById('predictBtn');
const uploadCircle = document.getElementById('uploadCircle');

if (fileInput) {
  fileInput.addEventListener('change', function(e) {
    const file = e.target.files[0];
    if (file) {
      predictBtn.style.display = 'inline-block';

      const reader = new FileReader();
      reader.onload = function(event) {
        uploadCircle.style.backgroundImage = `url(${event.target.result})`;
        uploadCircle.textContent = '';
      };
      reader.readAsDataURL(file);
    } else {
      predictBtn.style.display = 'none';
      uploadCircle.style.backgroundImage = 'none';
      uploadCircle.textContent = 'UPLOAD IMAGE';
    }
  });
}
