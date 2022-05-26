const uploadButton = document.getElementById("uploadButton");
const inputfile    = document.getElementById("inputfile");

uploadButton.addEventListener('click', async _ => {
  try {

    const file     = inputfile.files[0];
    const formData = new FormData();

    formData.append("file", file);

    uploadButton.disabled = true;
    uploadButton.style.opacity = 0.5;

    await fetch("http://127.0.0.1:5000", {
      method: 'POST',
      body: formData,
    });

    window.location.reload();

  } catch(err) {
    console.error(`Error: ${err}`);
  }

});