package ch.zhaw.javaproject;

import java.io.ByteArrayInputStream;
import java.io.File;
import java.io.InputStream;

import org.springframework.core.io.InputStreamResource;
import org.springframework.core.io.Resource;
import org.springframework.http.MediaType;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.multipart.MultipartFile;
import org.springframework.web.reactive.function.BodyInserters;
import org.springframework.web.reactive.function.client.WebClient;

@RestController
public class ClassificationController {

    private Inference inference = new Inference();

    @PostMapping(path = "/analyze-java")
    public String predictJava(@RequestParam("image") MultipartFile imageFile) throws Exception {
        System.out.println(imageFile);
        return inference.predict(imageFile.getBytes()).toJson();
    }


    @PostMapping(path = "/analyze-python")
    public String predictPython(@RequestParam("image") MultipartFile image) throws Exception {
        InputStream is = new ByteArrayInputStream(image.getBytes());
        var uri = "http://localhost:8080/predictions/fruitclassifier";
        if (this.isDockerized()) {
            uri = "http://model-serving:8080/predictions/fruitclassifier";
        }
        var webClient = WebClient.create();
        Resource resource = new InputStreamResource(is);
        var result = webClient.post()
                .uri(uri)
                .contentType(MediaType.MULTIPART_FORM_DATA)
                .body(BodyInserters.fromResource(resource))
                .retrieve()
                .bodyToMono(String.class)
                .block();
        return result;
    }

    private boolean isDockerized() {
        File f = new File("/.dockerenv");
        return f.exists();
    }

}