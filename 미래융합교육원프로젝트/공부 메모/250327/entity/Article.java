package com.example.firstproject.entity;

import jakarta.persistence.Column;
import jakarta.persistence.Entity;
import jakarta.persistence.GeneratedValue;
import jakarta.persistence.Id;
import lombok.AllArgsConstructor;
import lombok.ToString;

@AllArgsConstructor
@ToString
//엔티티 때문에 생성한 클래스이므로 엔티티 선언
@Entity
public class Article {
    
    @Id //엔티티 대표값
    @GeneratedValue // 자동 생성
    private Long id; // PK 
    
    @Column // DB
    private String title; // 일반 칼럼

    @Column // DB
    private String content; // 일반 칼럼

    
//    // Article 생성자 추가
//    public Article(Long id, String title, String content) {
//
//        this.id = id;
//        this.title = title;
//        this.content = content;
//
//    }
//
//    // toString() 메서드 추가
//    @Override
//    public String toString() {
//        return "Article{" +
//                "id=" + id +
//                ", title='" + title + '\'' +
//                ", content='" + content + '\'' +
//                '}';
//    }

    
}
