package com.example.firstproject.controller;

import com.example.firstproject.dto.ArticleForm;
import com.example.firstproject.entity.Article;
import com.example.firstproject.repository.ArticleRepository;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Controller;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PostMapping;

@Slf4j // 로깅 기능(print 대신)
@Controller
public class ArticleController {
    @Autowired
    private ArticleRepository articleRepository; // 의존성 주입

    // HTTP 프로토콜
    @GetMapping("/articles/new")
    public String newArticleForm(){
        return "articles/new";
    }


    @PostMapping("/articles/create") // 서브밋 매핑에 연결
    public String createArticle(ArticleForm form){ // ArticleForm 여기선 DTO
        // 인스턴스 제작 ArticleForm form
//        System.out.println(form.toString()); // 데이터 묶음
        log.info(form.toString());
        // 1. DTO를 엔티티 변환
        Article article = form.toEntity();
//        System.out.println(article.toString()); // 엔티티
        log.info(article.toString());

        // 2. 리파지터리로 엔티티 DB로
        Article saved = articleRepository.save(article); // save객체에 저장
//        System.out.println(saved.toString()); // DB
        log.info(saved.toString());
        return "";
    }

    // get 매핑 post 매핑
    // get과 post비교
    // get은 데이터를 가져오는 용도
    // 전송방식은 URL에 쿼리 문자열 형식으로 전송됨
    // URl에 포함되어 데이터 노풀, 보안 취약
    // 최대 데이터 길이가 제한적임

    // post는 데이터를 서버로 전송하는 목적, 아까 사용했던 form 데이터 전송 등
    // 전송하는 데이터가 URl에 포함되어 전송되는 방식이 아닌 body에 포함되어 전송됨
    // 데이터 제한이 get에 비해 훨씬 큰 데이터 가능(json 등)
    
    // HTTP 상태 코드
    // 200 ok 요청 처리 정상
    // 40x 잘못된 요청
    // 50x 서버 오류

    //

}
