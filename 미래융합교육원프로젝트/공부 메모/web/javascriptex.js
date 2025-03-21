function over(obj){
	// obj<매개변수 src
	obj.src="샴.jpg";
}	


function out(obj){
	obj.src="먼치킨.jpg";
}


// document.write("<h3>AAAAAAAAAA외부 스크립트 document.write로 작성</h3>");
// // head에 작성 시 맨 위로 / body에 작성 시 맨 아래로
// console.log("123 console.log 작성");
// // console.log < print 처럼 오류 점검
// let ret=prompt("이름","박");
// if (ret==null) {
// 	console.log("null case");
// }
// else if(ret==""){
// 	console.log("문자열 입력 없이 확인을 누른 경우(empty)");
// }
// else{
// 	console.log(ret);
// }
// 문법 형태가 자바와 같음

// alert("입력이 되지 않았습니다.");
// confirm("확실합니까?");
// confirm("질문 a b"); true/false

// js 변수 데이터 타입
// a_1 a1 aa A a
// 문장의 구분 ;
// 주석문 //
// 자바스크립트의 데이터 타입
// 숫자타입/논리타입/문자열타입/객체타입/null(참조 값이 없음)


// 변수 선언
// let / var /const 사용 가능
// var는 자바스크립트 언어 도입 시점부터 존재하던 변수 선언 타입
// ECMAscript es6 시점 이후에 let / const가 생겨남

// 변수의 범위

// 전역 변수: 함수 밖에서 선언하거나 var/let 키워드 없이 아무 곳에서 선언된 변수
// 전역 변수는 프로그램 실행시점부터 종료 시점까지 살아있는 변수이다.
// 프로그램 전역적으로 접근하여 사용할 수 있음


// 지역 변수
// 함수 내에서 var/let 키워드로 선언된 변수, 함수가 시작되면 생성되고 함수가 종료되면 소멸
// 함수 내에서만 사용 가능

// 블록 변수
// let키워드로 if/while/for 등 블록(스코프) 내에서 선언된 변수, 블록 내에서만 사용, 블록이 끝나면 사라짐
// 블록이 끝나면 사라지므로 블록 밖에서는 접근불가
// let으로 선언된 변수는 암시적으로 전역 객체에 속하지 않음(무조껀은 아님)/ 지역&블록 변수 만들 떄 적합

// let x;
// // 전역 변수 x선언 , var로 선언해도 동일
// function f(){ // 함수 f 정의
// 	let y; // 함수 내 지역 변수 y선언, var로 선언해도 동일
// 	x=10; // 전역 변수 x 10 초기화
// 	y=20; // 지역 변수 y 값 재할당
// 	z=30; // 전역 변수 z 선언 및 30 초기화(처음 생겨서)
// 	if(y==20){ // 지역 변수 y
// 		let b=40; // 블록 변수 b 선언 및 초기화
// 		b++; // 블록 변수 값 +1 처리
// 	}
// 	// 1. b제외 전부 접근 가능
// }
// // 2. x와 z 접근 가능



// var x=10; // 전역 변수 x 선언
// function xx(){
// 	var x; // 지역 변수 x 선언
// 	x=1; // 지역 변수 값 할당
// 	this.x=100; // 전역 변수 x에 100할당
// 	// 여기 this는 전역 변수
// }

//let으로 선언된 전역변수는 this로 접근 불가


//var로 선언된 전역 변수: var로 선언된 전역 변수는 전역 객체의 속성으로 저장된다.
// 브라우저 환경에서 전역 객체란? => window 객체
// BOM / DOM < 객체화?
// window.
// https://velog.io/@juhan/DOM%EC%9D%B4%EB%9E%80  < 윈도우 객체&DOM
// console.log(window.document);
// console.log(window.location);
// console.log(window.name);

// 여기선 html 사용 불가 객체화된 무언가를 사용/ 그거에 따라 작성

// var로 선언된 전역변수: var로 선언된 전역변수는 전역 객체의 속성으로 저장된다.
// 브러우저 환경애서 전역 객체란? -> window 객체이고 this로 접근, window과 this 같다
// var로 선언된 전역변수 window 객테 프로퍼티로 저장 -> this 가능

var globalvar=100;
console.log(this.globalvar);
console.log(this.location);

// var와 let의 작동상 차이점
// var는 동일한 변수를 재선언 할 수 있음
// let은 동일한 변수를 재선언 할 수 없음
// let은 변수의 사용범위를 블록으로 제한
// var은 제한 없음

// const는 상수로 변하지 않는 값(최종값)을 가지는 키워드
const MAX=10;
// const 상수는 대문자 식별자 이름
// 상수는 값 변경 x
// 상수는 재선언 x
// 상수는 선언된 블록 내에서만 사용


// 리터럴: 값
// 자바스크립트에서 리터럴 타입 종류
// 정수 8진수 10진수 16진수
// 실수
// 논리
// 문자열
// null: 값이 없음을 뜻
// NaN: 수가 아님을 뜻


// 연산자: 자바와 동일
// --a 
// a--

// 논리 or -> || / and -> && / not -> !




// 기본 문법은 for if while dowhile function 등 기본 문법은 자바와 동일하다


// 자바스크립트의 객체
// 1. 코어 객체: Array/Date/Atring/Math 코어객체 생성 시 new 키워드 사용 
// > import datetime 등
let today=new Date();
let msg=new String("안녕하세요");
// 객체에 대한 접근: .을 찍어서 접근
// 객체 프로퍼티와 메서드가 있음
// 자바스크립트에서는 객체의 소명 방법 없음
// 가비지컬렉터에 의해 자동 소명, 자동 관리 됨

// Array객체
let arr=["A","B"];
let n=[1,2,3];
// 인덱스로 접근 가능
arr[0]="C"; // 인덱스 통한 접근 후 값 변경
// for(초기값,어디까지,증감식) 자바와 동일
let a= new Array("1","2","3","4");
let b= new Array(7);
// 7칸 짜리 배열 제작
b[0]="1";
let c=new Array(); // 빈 배열 생성

// 배열의 length프로퍼티: 배열 요소 수 리턴
// 배열에는 여러 타입 데이터가 섞여 저장될 수 있음


// Array 객체의 주요 메서드

// concat() 배열 이어 붙이기
// join([sparator]) 배열의 모든 원소를 연결하여 하나의 문자열로 리턴
// reverse() 뒤집는다 
// slice() 자른다
// sort() 정렬한다
// toString() 요소를 모두 연결해서 문자열로 만들어 리턴


// Date객체
// Date객체는 시간 정보를 담는 객체
// new 키워드로 생성
// 실제 Date객체 생성된 시간 정보(코드가 실행된 시점)
// let d= new Date(2024,11,1); //24년도 11월 1일
// let now = new Date(); // 현 시점 객체 생성
// let date = now.getDate(); // 날짜 값 리턴
// let hour= now.getHours(); // 시간 값 리턴
// getFullYear()4자리 연도 리턴
// getMonth() 0~11 사이 월 값 리턴
// getDate() 한달 내 날짜 리턴 1~31
// getDay() 한주 내 요일 값을 정수 형태로 리턴 0=일요일 6=토요일
// getTime() UTC 기준 경과 초

// Math 객체
// Math 객체의 주요 메서드
// abs()절대값
// sin()
// cos()
// random(): 0~1 사이 임의 실수 리턴
// floor(x): x보다 작거나 같은 수 중 가장 큰 수 리턴
// round(x): x 반올림한 수
// sqrt(x): x의 제곱근 리턴

// eval()함수: 문자열로 구성된 수식의 연산 결과 리턴
// eva("7*7")하면 리턴 49옴

// random
// 사용 prompt("")

// prompt("랜덤A * 랜덤B");

// var / let
ra= Math.random();

// 2. DOM 객체

// document.getElementById();
// document.body.style.backgroundColor='red';
// 위 문장을 if문 내 실행문으로 활동해서 현재 시간이 짝수초면 바디를 빨간 색으로 홀수초면 파란색으로 표현하세요
// if 짝수

let now = new Date();
// 현재 날짜,시간 등의 값
let hour = now.getHours();
// 시간 값만
if(hour/2 == 0){
	console.log(hour)
	document.body.style.backgroundColor='red';
}else{
	console.log(hour)
	document.body.style.backgroundColor='#5ABEFF';
}
// 

// 3. BOM 객체
// browser object model 객체
// DOM 객체들은 HTML 태그들을 객체화 한 것이고
// BOM은 윈도우와 윈도우에 로드된 문서의 URL, 브라우저가 출력된 스크린 장치, 브라우저의 종류와 기능등
// 브라우저와 브라우저가 실행하고 있는 환경을 자바 스크립트코드로 접근하기 위해 객체화하는 것
// BOM에 속한 객체들은 HTML페이지의 내용과 관계없는 순수 브라우저 관련 객체들이다.


// BOM 종류
// 1. window
// 2. location
// 3. history
// 4. screen

// window 객체는 열려있는 브라우저 윈도우나 탭 윈도우를 나타내는 객체로 윈도우마다 하나의 window객체가 생성
// window 객체가 생성되는 경우는 다음 3가지
// 1. 브라우저가 새로운 웹 페이지를 로드할 때
// 2. iframe 하나 당 하나의 window객체이다
// 3. 개발자가 js코드를 통해 직접 window객체를 임의로 생성하는 종류 ex) window.open("url","윈도우명","속성")

// BOM은 w3c의 규격이 아니다
// window객체의 프로퍼티와 컬렉션,메서드 목록
// 프로퍼티와 컬렉션
// window: 현재 윈도우 객체에 대한 ref
// document: 윈도우에 담긴 document 객체에 대한 ref
// history: 히스토리 객체
// menubar: 메뉴바 객체
// locationbar: 위치바 객체
// length: 윈도우에 존재하는 iframe 수
// frames: 윈도우에 존재하는 iframe 객체들의 컬렉션 
// innerWidth: 수직 스크롤바를 포함한 브라우저 내 HTML문서가 출력되는 영역의 폭
// innerHeigt: 수평 스크롤바를 포함한 브라우저 내 HTML문서가 출력되는 영역의 넓이
// outerWidth: 윈도우 전체의 폭
// outerHeigt: 윈도우 전체의 넓이
// screenX: 스크린상 윈도우 x좌표
// screenY: 스크린상 윈도우 y좌표
// scrollX: 현재 문서의 스크롤 x 값
// scrollY: 현재 문서의 스크롤 y 값


// 메서드
// open(): 새 윈도우 오픈 
// close(): 윈도우 닫기
// focus(): 윈도우에 포커스를 줘서 키보드 입력 가능 상태로 만듬
// blur(): 윈도우가 포커스를 잃음
// stop(): HTML 페이지를 로딩 중단(브라우저의 stop 버튼 누른 것과 동일)
// alert(): 경고 다이얼로그 출력
// confirm(): 확인 다이얼로그
// prompt(): 입력 다이얼로그
// setInterval(): 타임아웃 코드가 반복 호출되도록 타이머 설정
// clearInterval(): 설정된 타이머 제거
// setTimeout(): 타임아웃 단발성 타이머 설정
// clearTimeout(): 단발성 타임아웃 해체
// moveBy(): 저장된 픽셀만큼 윈도우 이동
// moveTo(): 지정된 위치로 윈도우 이동
// resizeBy(): 저장된 크기만큼 윈도우 크기 조절
// resizeTo(): 저장된 크기로 윈도우 크기 조절
// scrollBy(): 주어진 픽셀만큼 상하좌우 스크롤 한다
// scrollTo(): 특정 좌표로 스크롤 한다

// window 객체에도 이벤트 리스너를 등록할 수 있다.

// window.open("https://www.naver.com/","_blank","width=300,height=300");

// url: 페이지의 주소, 이 매개변수는 필수, null 허용하지 않음
// 윈도우명(생략가능): 새로여는 윈도우의 이름 전달 매개변수 
//     _blank: 이름 없는 새 윈도우 
//     _parent: 새 윈도우를 열지 않고 현재 윈도우의 부모 윈도우에 웹페이지 출력
//     _self: 현재 윈도우에 페이지 출력 
//     _top: 브라우저 윈도우에 웹 페이지 출력
// 속성(생략가능): 윈도우의 모양이나 크기 등 속성을 설정하는 부분 ex) "resizable=yes","resizable=1"
//     width/height/left/top/location/menubar/scrollbars/status


let newWin=null;
function load(URL){
	newWin=window.open(URL,"_blank","left=300, top=300,width=300,height=300");
}

function closeNewWindow(){
	if(newWin==null || newWin.closed){
		return;
	}else{
		newWin.close();
	}
}

// window 타이머 활용

// window에는 타이머 작용 메서드가 있어 이를 활용하여 시간에 따라 웹 페이지에 동적인 효과를 줄 수 있음
// 단발성 -> setTimeout(): 메서드에 밀리포 단위 타임아웃 지연시간과 타임아웃 코드를 작성한다
function myAlert(t){
	alert(t+"초 지남");
}

// let timerID=setTimeout("myAlert(3)",3000);
// clearTimeout(timerID); //timeout은 단발성이기에 수동으로 clearTimeout으로 끄는 경우가 적음

// // 반복성 -> setInterval()

// let timeID2=setInterval("myAlert(3)",3000);
// clearInterval(timeID2); //interval은 무한 반복이기에 clearInterval함수로 꺼줘야 함

// 실습> setTimeout()을 통해서 5초간 div박스에 마우스를 올려놓고 유지하면 새로운 윈도우가 하나 열리도록 해본다
// function f(){ // 2초마다 호출할 함수의 정의

// }
// let timeId=setInterval('f()',2000); //on
// clearInterval(timeId); //off
// setTimeout("load",3000ms);
// let boxA=document.getElementsByClassName("setBox")[0];
// boxA.addEventListener('',openEvent);
// function openEvent(){
// 	setTimeout(function(){

// 	});
// }

////////////////////////////예제 
// let TimerID3 = null;
// function mouseOver(){
//    TimerID3=setTimeout("load('https://www.google.com/')",3000);
// }
// function mouseOut(){
//    clearTimeout(TimerID3);
// }
// setTimeout(function(){
// 	// let boxA=document.getElementsByClassName("setBox")[0];
// 	// load('https://www.naver.com/') / onmouseleaver
// 	$('.boxA').hide()},3000);

// p태그에 어떤 문장이 쓰여있고 p태그의 기본 배경 색상은 red이다
// p태그 우측에는 2개의 div로 만들어진 버튼이 있고 첫번째 버튼은 ON이고 두번쨰 버튼은 OFF버튼임
// ON버튼을 누르면 2초마다 반복적으로 p태그의 배경색상이 R>G>B>R>G...형태로 변경되는 작동이 켜지고 
// OFF버튼을 누르면 자동이 꺼진다.

// let TimeId = null;
//////////예시
backcolor=[
	"blue","skyblue","lightpink"
]
let index =0;

let colorp=document.getElementById('aa');
let btnA=document.getElementById('onA');
let btnB=document.getElementById('offA');
let TimeId; // 선언 필수 


function backcolorA(){
	// document.body.style.backgroundColor='red';
	// let ranColor=Math.floor()
	colorp.style.backgroundColor=backcolor[index];
	index++;
	if (index>=backcolor.length) {
		index=0;
	}
}

// function timeon(){

// }

btnA.onclick=()=>{
	TimeId=setInterval('backcolorA()',1000);
}

btnB.onclick=()=>{
	clearInterval(TimeId);
}

// function timeout(){
// 	clearInterval(TimeId);
// }

// 4. window 객체


// 44일차 반응형
// var btn=document.getElementById('addtodo');
// var btn=document.getElementById('newtodo');
// var btn=document.getElementById('');

// todolist.addEventListener('click',(event)=> { //매개변수로 마우스 이벤트(포인터 이벤트 객체 전달 받음)
// 	console.log(event); //이벤트 객체는 해당 이벤트가 발동된 target과 여러 속성을 가지고 있다.
// 	if(){
// 		// 이벤트 
// 	}
// })



// 이벤트와 이벤트 객체
// 이벤트란 마우스/키보드 입력, 이미지나 문서의 로딩, 타임아웃 등 사용자의 입력 외 문서나 브라우저의 상태 변화를 브라우저가 자바스크립트에게 알리는 notification.

//이벤트 리스너란 
// 발생한 이벤트에 대한 적정한 대처를 하기 위해 작성된 자바스크립트 코드

//주요 이벤트 리스너 목록
// onkeydown: 키보드의 아무 키를 누르는 순간
// onkeypress: 키보드의 알파뉴메릭(알파벳과 숫자만 허용) 키 누르는 순간
// onkeyup: 키보드 누른 키 놓는 순간

// onclick: 좌클릭
// oncontextmenu: 우클릭
// ondbclick: 더블 클릭
// onmousedown: 마우스의 물리버튼 누르는 경우
// onmouseover: 마우스 커서가 객체 영역 안으로 들어오는 순간
// onmouseout: 마우스 커서가 객체 영역 밖으로 나가는 순간
// // onmouseenter: 마우스 커서가 객체 영역 안으로 들어오는 순간(자식 객체에게 전파)
// //  onmouseleave
// onmouseleave : 마우스 커서가 객체 영역 안으로 들어오는 순간(자식 객체에게 전파)
// onmouseout: 마우스 커서가 객체 영역 밖으로 나가는 순간(자식 객체에게 전파)
// onmouseup: 누른 마우스 버튼 놓는 순간
// onwheel: 마우스 휠 변화가 생길 떄 

// onabort: 이미지나 문서의 로딩이 중단되는 경우
// onerror: 문서나 이미지,리소스 등의 로딩에서 오류가 발생할 때
// onload: 문서나 이미지의 로딩이 완료된 후

// onresize: 윈도우,프레임 혹은 객체 크기가 변경될떄

// onfocus: 객체가 포커스를 가질 떄
// onchange: 체크박스, input 등의 내용이 변하는 경우
// onreset: 폼이 reset 되는 경우
// onsearch: input type search 창에 엔터 입력시
// onselect: input 영역의 텍스트를 선택할 때
// onsubmit: 제출버튼 클릭시 발생



// 이벤트 리스너 만드는 방법
// 44일차 주사위 제작함

// 1. HTML태그에 직접 작성하는 방법
// <p onclick="">p태그입니다.</p>


// 2. DOM객체의 이벤트리스너 프로퍼티에 작성하는 방법
// vat pp=document.getElementById('pp'); dom객체화 // pp.a()// pp.a
// founctionx(){}; // 함수 선언
// pp.onmouseover=x; //dom객체의 이벤트리스너 프로퍼티에 함수를 등록 //등록하는 경우에는 함수만 / 출력까지면 () 추가 


// 3. DOM객체의 addEventListener() 메서드로 등록하는 방법
// addEventListener(eventName,listener,[,useCapture])함수의 매개변수 [선택적으로 넣을 수 있는 것]
// eventName:이벤트 타입을 나타내는 문자열 click,load,keydown 등
// listener: 이벤트 리스너를 등록할 함수 이름
// useCapture: true로 놓는 경우 이벤트 흐름 중 캡쳐 단계에서 실행될 리스너로 등록하는 것,
// false로 등록하면 버블 단계에서 실행되는 리스너로 등록함, 기본값은 false

// pp.addEventListener("click,x") // 정의되어 있는 (익명)함수 등록
// pp.addEventListener('click',fuction(){함수의 실행문}) // 익명함수 사용
// pp.addEventListener('click',()=>{}) // 44일차엔 이런식으로 적용 (화살표 함수)


// 이벤트 객체란
// 이벤트가 발생하면 브라우저는 발생한 이벤트에 대한 여러 정보를 담은 이벤트 객체를 만들어서 이벤트 리스너에게 전달한다. 
// onmenusedown 이벤트의 경우 마우스 관련이기 때문에 마우스의 좌표 등 정보를 담은 마우스 이벤트 객체에 전달하고 
// onkeydown의 경우 키보드 관련으로 키보드의 어떤키가 눌렸는지 keycode등의 정보를 담아 전달한다.

// 이벤트가 처리되고 나면 이벤트 객첸는 소멸한다.
// 브라우저 한 개의 이벤트를 완전히 처리한 후 다음 이벤트를 처리하므로, 
// 이벤트 리스너 실행 중에는 오직 한개의 이벤트 객체만 존재한다.

// 마우스 이벤트 객체 and 키보드 이벤트 객체를 각각 console.log

// 이건 내가 한거
// var dd=document.getElementById('id1');

// dd.addEventListener('click',()=>{
// 	console.log(dd);
// });

// 이건 예시
// document.body.addEventListener('keypress',(event)=>{console.log(event)})
// // 키보드 입력 시 로그
// document.body.addEventListener('wheel',(event)=>{console.log(event)})
// 마우스 휠 이동 시 로그


// 이벤트 객체의 target프로퍼티
// 타겟 프로퍼티는 이벤트의 대상을 가르킴
// 이벤트 타겟이란 이벤트를 유발시킨 객체를 의미함
// target과 유사함 프로퍼티로 currentTarget프로퍼티가 있다.
// currentTarget은 이벤트가 흘러가는 경로 상에 있는 DOM 객체 중 현재 이벤트 리스너를 실행하고 있는 DOM객체를 가르킨다.

// var ppp=document.getElementById('d11');
// ppp.addEventListener('click',ff);
// function ff(e){
// 	let text="type"+e.type+"<br>"+"target"+e.target+"<br>"+"currentTarget"+e.currentTarget+"<br>";
// 	// 하나의 문자열 이지만 target / currentTarget 차이 확인
// 	ppp.innerHTML=text;
// }

// type 프로퍼티는 현재 발생한 이벤트의 종류
// target 프로퍼티는 이벤트가 실제로 발생한 요소(직접 누른 곳)
// currentTarget 프로퍼티는 이벤트가 바인딩 된 요소(선언)
// defaultPrevented프로퍼티는 이벤트의 디폴트 행동이 취소되었는지 여부 t/f로 나타남
// preventDefault() 메서드는 이벤트의 디폴트 행동을 취소시키는 메서드


// HTML캐그 중 몇몇은 특정 이벤트에 대해 디폴트 행동을 한다.

// 디폴트 행동의 예시
// a링크를 클릭하면 이동하는 것
// submit 버튼을 클릭하면 폼 데이터를 전송하는 것
// reset을 입력하면 폼이 초기화 되는 것
// wheel을 굴리면 브라우저 스크롤이 움직이는 것

// 이벤트 객체의 cancelable 프로퍼티가 true인 경우만 preventDefault가 가능하다


// 이벤트의 흐름
// 이벤트가 발생하면 이벤트는 타겟 객체에 전달한다

// 이벤트가 단번에 타겟 객체로 직접 전달되는 것이 아니라 
// windowo객체로부터 DOM트리를 타고 중간 DOM객체들을 거쳐 타겟 객체로 이벤트가 흘러들어가고, 
// 다시 반대 방형으로 이동하여 window 객체에 도달한 후 이벤트는 소멸한다. 
// 위 과정을 이벤트의 흐름이라고 한다.

// 이벤트의 흐름은 2가지 단계로 나누어 볼 수 있다. 1. 캡쳐단계, 2.버블 단계

// 캡쳐 단계는 window 객체에서 타겟 객체까지 전파되는 과정
// 버블 단계는 타겟 객체에서 거꾸로 window까지 이벤트 객체가 전파되는 과정
// DOM객체들은 동일한 이벤트에 대해 캡쳐 리스너와 버블 리스너를 모두 가질 수 있음

// function cap(){
// 	console.log(1);
// }

// function bub(){
// 	console.log(2);
// }

// var bt1=document.getElementById('bt1');

// bt1.addEventListener('click',cap,true);
// bt1.addEventListener('click',bub,false);

// 캡처와 버블 단계 구분 : 이벤트 발동 순서에 대한 관리 / 이벤트 우선 순위 관리 필요할 때

// 이벤트 객체의 멤버 중 이벤트 흐름과 관계된 멤버
// cancelable : 디폴트 취소 여부
// stopPropagation() : 객체에 등록된 리스너를 모두 실행 후 이벤트 흐름 중단
// stopImmediatePropagation() : 현재 리스너만 실행하고 이벤트 흐름 즉각 중단


// 마우스 핸들링
// 마우스 객체 관련 프로퍼티
// x,y : x,y는 타겟 객체의 부모 객체 내애서 마우스 좌표
// clientX,clientY : 브라우저 윈도우의 문서 출력 영역 내에서 마우스 좌표
// screenX,screenY : 스크린을 기준으로 한 마우스 좌표
// offsetX,offsetY : 타겟 객체 내애서의 마우스 좌표
// button : 눌러진 마우스 버튼 종류 > 0: 아무것도 안눌린 상태, 1: 왼쪽 버튼, 2: 오른쪽 버튼, 3: 왼쪽,오른쪽 모두, 4: 중간버튼
// wheelDelte : 휠이 구른 방향, 양수:위쪽으로 굴림 / 음수: 아래쪽으로 굴림

// document.body.addEventListener('wheel',(event)=>{console.log(event)})


// 키보드 이벤트 객체 프로퍼티
// code : 눌려진 키의 이름
// key : 눌려진 키의 문자열
// altkey : 알트키 눌린 상태 여부
// ctrlkey : 컨트롤키 눌린 상태 여부
// shiftkey : 쉬프트키 눌린 상태 여부
// keycode : 키코드 값 숫자


// 3*3
// let tds;
// let prevIndex=0;
// let index=0;
// window.onload=function(){
// 	tds=document.getElementsByTagName("td");
// 	tds[index].style.backgroundColor="orchid";
// }

// window.onkeydown = function(e) {
// 	switch(e.key){
// 	case "ArrowDown":
// 		if(index/3>=2) return;
// 		index+=3;
// 		console.log(index);
// 		break;	
// 	case "ArrowUp":
// 		if(index/3<1) return;
// 		index-=3;
// 		console.log(index);
// 		break;	
// 	case "ArrowLeft":
// 		if(index%3==0) return;
// 		index--;
// 		console.log(index);
// 		break;	
// 	case "ArrowRight":
// 		if(index%3==2) return;
// 		index++;
// 		console.log(index);
// 		break;

// 	}

// 	tds[index].style.backgroundColor="orchid";
// 	tds[prevIndex].style.backgroundColor="white";
// 	prevIndex=index;

// }

// 테이블 누르면 카드 뒤집히는 거
// 4*4 테이블로 만들 td칸 각 칸은 누르면 뒤집힌다. 뒤집힌 칸에는 이미지가 뜬다. 이미지 종류는 8개

// let tds;
// let prevIndex=0;
// let index=0;
// window.onload=function(){
// 	tds=document.getElementsByTagName("td");
// 	// tds[index].style.background="url(https://cdn.metavv.com/prod/uploads/thumbnail/images/10043263/167100535142741_md.png) no-repeat";
// 	tds[index].style.borderColor='plum';
// }

// window.onkeydown = function(e) {

// 	// console.log(key);
// 	switch(e.key){
// 	case "ArrowDown":
// 		if(index/4>=3) return;
// 		index+=4;
// 		console.log(index);
// 		break;	
	
// 	case "ArrowUp":
// 		if(index/4<1) return;
// 		index-=4;
// 		console.log(index);
// 		break;	
	
// 	case "ArrowLeft":
// 		if(index%4==0) return;
// 		index--;
// 		console.log(index);
// 		break;	
	
// 	case "ArrowRight":
// 		if(index%4==3) return;
// 		index++;
// 		console.log(index);
// 		break;
	// case "Enter":
	// 	tds[index].classList.toggle("effectCSS");
	// 	// tds[index].classList.add('effectCSS'); //수동
	// 	// tds[index].classList.remove('effectCSS'); //위에랑 세트
	// 	if((index == 8)||(index == 13)){
	// 		tds[index].style.background='plum';
	// 	}else if((index == 5)||(index == 6)){
	// 		tds[index].style.background='skyblue';
	// 	}else if((index == 0)||(index == 4)){
	// 		tds[index].style.background='lightgreen';
	// 	}else if((index == 1)||(index == 10)){
	// 		tds[index].style.background='lightcoral';
	// 	}else if((index == 3)||(index == 15)){
	// 		tds[index].style.background='lightpink';
	// 	}else if((index == 2)||(index == 9)){
	// 		tds[index].style.background='lightyellow';
	// 	}else if((index == 11)||(index == 7)){
	// 		tds[index].style.background='midnightblue';
	// 	}else if((index == 14)||(index == 12)){
	// 		tds[index].style.background='lightsteelblue';
	// 	}
		// if (index > 0){
		// 	if (index == 8){
		// 		tds[index].style.background='plum';
		// 	} else if (index ==13) {
		// 		tds[index].style.background='plum';
		// 	}
		// }
		
		// if ((index == 8)&&(index == 13)){
		// 	tds[index].style.background='plum';
		// } 이중 if
		// 만약 둘다 참일 때는 유지 그렇지 않을 경우는 해체
// 		break

// 	// case "Enter":
// 	// 	if(index == 1) {
// 	// 		tds[index].style.effectCSS;
// 	// 		break;
// 	// 	}


// 	}

// 	// tds[index].style.background="blue";
// 	// tds[prevIndex].style.background="white";
// 	tds[index].style.borderColor='plum';
// 	tds[prevIndex].style.borderColor='blue';
// 	prevIndex=index;
// 		// event.preventDefault();
// }

// tds[0].addEventListener();
// tds[index].addEventListener('click',function(){

// })



	// case "Enter":
	// 	if (index == 0) {
	// 		tds[prevIndex].style.background="red";
	// 		tds[index].style.background="red";
	// 		console.log(index);
	// 	}if (index == 15) {
	// 		tds[prevIndex].style.background="red";
	// 		tds[index].style.background="red";
	// 		console.log(index);
	// 	} else {
	// 		console.log(index);
	// 	}


// //<div><span id="sp1" style="background-color: lightseagreen;">123456텍스트</div>
// let span=document.getElementById('sp1');
// let timerITV=setInterval("dort()",200);
// span.onclick=function (e){
// 	clearInterval(timerITV);
// }
// function dort(){
// 	let str=span.innerHTML;
// 	let firstChar=str.substr(0,1);
// 	let remains=str.substr(1,str.length-1);
// 	str=remains+firstChar;
// 	span.innerHTML=str;
// }

// window.scrollBy(0,10);
// window.scrollBy(10,-15);
// window.scrollTO(0,200);

// location 객체
// window.location
// location 객체는 윈도우에 로드된 웹페이지 url정보 나타내는 객체
// 윈도우가 열릴 때 자동 생성 됨
// window는 생략 가능하다


// https://www.example.com:8080/content/ABC.html#label1 < ex예시

// https: 프로토콜
// /www.example.com: 호스트 네임
// 8080: 포트 번호
// content/ABC.html : path(경로)
// #label1: 해시(#을 포함하는 앵커의 이름 문자열)


// history 객체
// 히스토리 객체는 사용자가 방문한 웹 페이지의 리스트 정보를 담고 있는 객체
// window.history 혹은 history로 접근

// history관련 프로퍼티와 메서드
// length: 히스토리 리스트에 있는 엔트리 수
// back(): 히스토리에 있는 이전 웹 페이지로 이동하는 메서드 브라우저 뒤로 버튼과 동일
// forward(): 히스토리에 있는 다음 웹 페이지로 이동하는 
// go(n): 히스토리에서 현재 웹페이지에서 

// let canvas=document.getElementById('cv1');
// let context=canvas.getContext("2d");
// getContext(2d)가 리턴한 context객체는 2차원 그래픽에 사용되는 프로퍼티와 메서드 제공
// let context 변수에 할당 된 객체는 CanvasRenderingContext2D 타입의 객체임
// HTML5의 캔버스는 비트맵 렌더링 엔진을 장착하여 브라우저에 즉시 그려내는 구조
// 속도가 빠르다
// canvas 객체의 프로퍼티와 메서드
// width 캔퍼스의 폭
// height 캔버스의 높이
// style 캔버스의 스타일 
// getContext(): 캔버스에 그림을 그리는 컨텍스트 객체 리턴하는 메서드

// context.beginPath(); // 그리기 시작
// context.strokeStyle='blue'; //색상
// context.rect(60,60,50,50);  //60,60좌표에서 50X50 사이즈 사각형
// context.stroke(); // 그리기

// context 객체 프로퍼티 메서드
// strokeStyle 선 색상
// fillStyle: 채우기 색상
// linewidth: 선 두께
// font: 텍스트 폰트 설정
// textAlign: 텍스트 정렬 방식

// beginPath(): 이전 경로 모두 지우고 새로운 경로 시작
// closePath(): 경로의 끝 점과 시작점을 연결하는 선을 추가하고 닫음 
// moveTo(): 경로에서 새 점을 만듦
// lineTo(): 경로의 끝 점과 연결하는 선을 추가한다
// rect(): 사각형 그리기
// arc(): 원호 그리기
// fill(): 경로에 만들어진 도형을 외각선 없이 내부 색을 채워 그린다
// stroke():  경로에 담겨진 모든 도형을 외각선만 캔퍼스에 그린다
// clearRect(): 캔퍼스에서 사각형 영역을 지운다
// fillRect(): 캔퍼스에서 외각선 없이 사각형 영역을 색으로 바로 칠한다
// strokeRect(): 사각형을 경로에 추가하지 않고 캔퍼스에 바로 그림
// strokeText(): 텍스트를 캔퍼스에 바로 그린다
// drawImage(): 이미지를 캔퍼스에 바로 그린다

// let canvas=document.getElementById('cv2');
// let context=canvas.getContext('2d');

// context.beginPath(); //빈 경로 변경하기
// context.strokeStyle="blue"; // 선 색상 설정
// context.rect(30,30,50,50); // 30,30에 50*50짜리 사각형을 경로에 넣는다
// context.stroke(); // 경로에 잇는 모든 도형의 외각선 그리기

// context.beginPath()
// context.fillStyle='violet';
// context.rect(60,60,50,50);
// context.fill();

// context.font='20px Gothic';
// context.fillStyle="green";
// context.fillText('text112233',100,50);




// 도형 그리기 
// 1.Path 만들기 
// 2. 캔퍼스에 담긴 경로를 그려내기

// 경로Path란 캔버스 위에 펜이 지나가는 길
// 경로 만들기는 캔버스에 그리고자 하는 도형들을 먼저 컨텍스트 내의 경로에 담는 과정이다
// moveTo() lineTo() rect() arc() 등 메서드는 켄버스에 직압 그리지 않고 도형을 경로에 추가하는 메서드다
// stroke이나 fill메서드가 호출될 때 실질적으로 경로에서 그림으로 캔퍼스에 올라옴
// closePath: 경로 종료

// context.beginPath();
// context.moveTo(120,20); // 경로에 담긴 도형은 그대로 두고 점 x,y를 새 시작점으로 삽입
// context.lineTo(20,50); // 경로의 끝점에서 x,y까지 직선을 경로에 추가
// context.lineTo(150,120);
// context.lineTo(120,20);
// context.strokeStyle="width";
// context.stroke();


// 원 그리기 arc()

// arc(x,y,radius,startAngle,endAngle,anticlockwise);
// x,y 원호의 중심
// radius 반지름
// startAngle: 시작 각도 3시가 기준점(시계 방향)
// endAngle: 종료 각도
// anticlockwise: 시계 방향은 true 반시계 방향은 false로 설정

// 각도법에서는 360도로 표현 -> 호도법으로 표현하면 라디안 단위로 표현
// 360도 = 2파이
// 1도 = 파이/180


// arc() 메서드에서 startangle과 endangle는 원주율로 지정한다
// 각도는 3시에서 기준점 시작
// math클래스에 있는 PI상수ㅡㄹ 이용해서 각도 설정
// 270도 = > Math.PI * 1.5
// 360도 => Math.PI * 2


// let canvas=document.getElementById('cv1');
// let context=canvas.getContext("2d");
// context.beginPath();
// context.strokeStyle='blue';
// context.arc(50,50,10,0,1.5*Math.PI,false);
// context.stroke();

// context.beginPath();
// context.strokeStyle='red';
// context.arc(100,70,50,Math.PI/2,Math.PI,true);
// context.stroke();


// let canvas=document.getElementById('cv1');
// let context=canvas.getContext("2d");


// context.fillStyle='violet';
// context.fillRect(20,20,100,100);

// context.beginPath();
// context.rect(20,150,100,100);
// context.fillStyle='violet';
// context.fill();

// context.strokeStyle="white";
// context.lineWidth=10;
// context.stroke();

// context.beginPath();
// context.moveTo(80,340);
// context.arc(80,340,50,0,1.5*Math.PI);
// context.closePath();

// context.fillStyle='blue';
// context.fill();

// context.strokeStyle='white';
// context.lineWidth=10;
// context.stroke();


// 이미지 그리기
// 이미지 객체의 생성
// 캔버스에 이미지를 그리기 위해서는 이미지를 담을 이미지 객체가 필요함
// 이미지 객체는 JS코드로 new 키워드 사용해서 만듬
// let img=new Image();

// 이미지 로딩과 onload (로딩 완료 후)
// 이미지 파일로부터 이미지를 로딩 시키기 위해서는 Image객체의 src프로퍼티 이용

// img.onload=function(){
// 	-----

// 	----- //img 객체에 로드된 이미지를 그리는 코드
// }

// img.src="~~~~"; //img객체에 ~~~~파일로부터 이미지 로딩 시작

// img.src='a.png'라인은 이미지 로딩을 시작시키는 코드
// 이미지 로딩이 완료되어야 이미지를 그릴 수 있기 때문에, 이미지 로딩을 지시하기 전 
// img.onload 리스너에 이미지가 로딩되었을 때 이미지를 그리는 JS코드를 쓴다


// 이미지 그리기
// 이미지 로딩이 완료되면 컨텍스트 객체의 drawImage() 메서드를 이용하여 이미지를 그린다
// drawImage(img,dx,dy)
// img: 이미지 객체
// dx: 이미지 그릴 캔버스 좌표 x
// dy: 이미지 그릴 캔버스 좌표 y
// width: 이미지 폭
// height: 이미지 높이

// let canvas=document.getElementById('cv1');
// let context=canvas.getContext("2d");
// let canvasTF=false;
// let startX;
// let startY;

// let img=new Image();
// img.onload=function(){
// 	context.drawImage(img,0,0,canvas.width,canvas.height);

// }
// img.src="https://upload3.inven.co.kr/upload/2020/04/24/bbs/i14579851604.jpg";

// 클릭을 하며 그릴 수 있도록 마치 그림판처럼
// canvas.addEventListener('mousemove',function(e){down(e)},false); 
// canvas.addEventListener('mousedown',function(e){down(e)},false); 
// // canvas.addEventListener('mousedown',function(e){draw(e)},false);
// function down(e){ //누르는 것(시작 위치)/ 마우스 좌표 설정
// 	startX=e.offsetX;
// 	startY=e.offsetY;
// 	// console.log(startX); //그림 그리기가 가능한 상태 마우스 위치
// }

// function up(){ // 마우스가 움직여도 그림 안그려지게

// 	// 만약 마우스 클릭했다면 true / draw 호출
// 	// if (canvasTF==false) {
// 	// 	return;
// 	// }
	
// }
// function out(){ //캔버스 영역 밖으로 나간 상태
// 	canvasTF=false;
// }

// function draw(){ //마우스 좌표 움직임에 따라 캔버스에 그림 그리는 함수
// 	// canvas.addEventListener('mousemove',function(e){down(e)},true); // 마우스 움직임대로
// 	context.lineTo(down(startX,startY)); // 마우스 위치대로 그림 , 
// 	console.log(startX); //그림 그리기가 가능한 상태 마우스 위치
// 	contex.stroke(); // 
// 	canvasTF=true; //그려지게
// }

// // // 예시 ()
let canvas=document.getElementById('cv1');
let context=canvas.getContext("2d");

let btn=document.getElementById('btnCan');

let canvasTF=false;
let x;
let y;


canvas.addEventListener('mousemove',(e)=>{
	x=e.offsetX;
	y=e.offsetY;

	if(canvasTF){
		context.lineTo(x,y);
		context.stroke();
	}else{
		console.log('클릭');
	}
})

canvas.addEventListener('mousedown',(e)=>{
	canvasTF=true;
	context.beginPath();
	context.moveTo(x,y);
})

canvas.addEventListener('mouseup',(e)=>{
	canvasTF=false;
	context.closePath();
})
canvas.addEventListener('mouseout',(e)=>{
	canvasTF=false;
	context.closePath();
	context.clearRect(0,0,canvas.width,canvas.height); //캔버스를 벗어나면 자동으로 지워짐(좌표값 초기화)
})
// 지우개 > 색 배경색으로 변경 

// btn.addEventListener('click',()=>{
// 	context.strokeStyle='lightblue';
// })

// :after는 가상셀렉터로 x:after는 x요소 뒷부분 의미
// x:after CSS셀렉터 내용에 content를 통해 x요소에 붙일 컨텐츠 작성
// display:table은 테이블처럼 보여지는 디스플레이
// clear:both는 float속성에 left/right로 배치한 요소와 다음 요소가 겹치지 않게 
// opactity는 투명도 속성 0~1
// user-select:none은 해당 요소의 텍스트를 드레그해서 선택 못하도록 설정하는 것


