package ch.zhaw.javaproject;

import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.multipart.MultipartFile;


@RestController
public class ClassificationController {

    private Inference inference = new Inference();

    @PostMapping(path = "/analyze")
    public String predict(@RequestParam("image") MultipartFile imageFile) throws Exception {
        System.out.println(imageFile);
        return inference.predict(imageFile.getBytes()).toJson();
    }

}