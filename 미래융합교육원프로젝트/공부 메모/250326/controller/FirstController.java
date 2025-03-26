package com.example.firstproject.controller;

import org.springframework.stereotype.Controller;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.ui.Model;

@Controller // 컨트롤러 선언
public class FirstController {

    @GetMapping("/hi") // URL 설정 , 한글 설정 시 글씨 깨짐 > application.properties 설정
    public String niceToMeetYou(Model model){ //메서드 작성
        model.addAttribute("username", "머스테치");
        return "greetings"; //greetings 파일 확인
    }

    @GetMapping("/bye")
    public String seeYouNext(Model model){
        model.addAttribute("username","머스테치");
        return "goodbye";
    }

    @GetMapping("/random-quote")
    public String randomQuote(Model model){
        String[] quotes = {
                ""+"",
                ""+"",
                ""+"",
                ""+""
        };
        int randomInt = (int) (Math.random() * quotes.length);
        model.addAttribute("randomQuote", quotes[randomInt]);
        return "quote";
    }
    
}
