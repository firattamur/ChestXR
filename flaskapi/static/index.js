const uploadButton = document.getElementById("uploadButton");
const inputfile    = document.getElementById("inputfile");

uploadButton.addEventListener('click', async _ => {
  try {

    let file     = inputfile.files[0];
    let formData = new FormData();

    formData.append("file", file);

    const response = await fetch("http://127.0.0.1:5000", {
      method: 'post',
      body: formData,
    });

    console.log('Completed!', response);

  } catch(err) {

    console.error(`Error: ${err}`);

  }

});