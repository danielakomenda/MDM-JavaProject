<script>
  let files;
  let previewSrc;
  let answerText = "";
  let isVisible = false;
  let analyzeView = false;
  let results = [];
  let likely_results = [];
  let unlikely_results = [];

  function checkFiles(event) {
    analyzeView = false;
    const selectedFiles = event.target.files;
    console.log(selectedFiles);

    if (selectedFiles.length !== 1) {
      alert("Bitte genau eine Datei hochladen.");
      return;
    }

    const fileSize = selectedFiles[0].size / 1024 / 1024; // in MiB
    if (fileSize > 10) {
      alert("Datei zu groß (max. 10MB)");
      return;
    }

    isVisible = true;
    files = selectedFiles;
    previewSrc = URL.createObjectURL(files[0]);
  }

  async function analyzeJava(file) {
    const formData = new FormData();
    formData.append("image", file);

    try {
      const response = await fetch("/analyze-java", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        throw new Error("Failed to upload and classify the image");
      }

      results = await response.json();
      // Reset arrays to ensure Svelte detects changes
      likely_results = [];
      unlikely_results = [];

      for (const result of results) {
        if (result.probability > 0.2) {
          likely_results = [...likely_results, result];
        } else {
          unlikely_results = [...unlikely_results, result];
        }
      }
      analyzeView = true;
      setTimeout(() => {
        const resultSection = document.getElementById("result-section");
        if (resultSection) {
          resultSection.scrollIntoView({ behavior: "smooth" });
        }
      }, 0);
    } catch (error) {
      console.error("Error:", error);
      answerText = "Error processing the image.";
    }
  }

  async function analyzePython(file) {
    const formData = new FormData();
    formData.append("image", file);

    try {
      const response = await fetch("/analyze-python", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        throw new Error("Failed to upload and classify the image");
      }

      results = await response.json();
      // Reset arrays to ensure Svelte detects changes
      likely_results = [];
      unlikely_results = [];

      for (const result of results) {
        if (result.probability > 0.2) {
          likely_results = [...likely_results, result];
        } else {
          unlikely_results = [...unlikely_results, result];
        }
      }
      analyzeView = true;
      setTimeout(() => {
        const resultSection = document.getElementById("result-section");
        if (resultSection) {
          resultSection.scrollIntoView({ behavior: "smooth" });
        }
      }, 0);
    } catch (error) {
      console.error("Error:", error);
      answerText = "Error processing the image.";
    }
  }
</script>

<div class="component">
  <div class="container-fluid">
    <h1>Fruit-Classifier</h1>
    <h5>Classifier created with DeepJavaLibrary (DJL)</h5>

    <p>Upload a picture of one of the following fruits:</p>
    <ul>
      <li>Apple</li>
      <li>Banana</li>
      <li>Cherry</li>
      <li>Grapes</li>
      <li>Kiwi</li>
      <li>Mango</li>
      <li>Orange</li>
      <li>Strawberry</li>
    </ul>

    <form>
      <div class="form-group">
        <label for="exampleFormControlFile1">Only .jpeg and .png are accepted</label>
        <input
          type="file"
          accept="image/png, image/jpeg"
          class="form-control-file"
          id="image"
          name="image"
          on:change={checkFiles}
        />
      </div>
    </form>

    {#if isVisible}
      <div>
        <p><b>Analyzed Image:</b></p>
        <img src={previewSrc} width="300" alt="preview" />
      </div>
      <button type="button" on:click={() => analyzeJava(files[0])}>Analyze with Java</button>
      <button type="button" on:click={() => analyzePython(files[0])}>Analyze with Python</button>
    {/if}

    {#if analyzeView}
      {#if results.length > 0}
        <div class="answer">
          <p class="answer-title">This fruit is most likely one of the following:</p>
          { #if likely_results.length<1}
                <p>This object doesn't seem to be one of the mentioned fruits.</p>
            {:else}
                {#each likely_results as result}
                    <tr>
                    <td>{result.className}:</td>
                    <td>{result.probability.toFixed(3)}</td>
                    </tr>
                {/each}
            {/if}
        </div>

        <div class="answer">
          <p class="answer-title">This fruit is most likely NOT one of the following:</p>
          {#each unlikely_results as result}
            <tr>
              <td>{result.className}:</td>
              <td>{result.probability.toFixed(3)}</td>
            </tr>
          {/each}
        </div>
      {/if}
    {/if}
    <div id="result-section"></div>
  </div>
</div>

<style>
  .answer {
    margin-top: 15px;
    margin-bottom: 5px;
  }
  .answer-title {
    font-weight: bolder;
  }
</style>
