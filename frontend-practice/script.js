// GSAP 플러그인 활성화
gsap.registerPlugin(ScrollTrigger);

// .box 요소에 애니메이션 적용
gsap.to(".box", {
    // 1. 스크롤 트리거 설정
    scrollTrigger: {
        trigger: ".animation-section", // 이 섹션이 보일 때 시작
        start: "top center",          // 섹션의 상단이 화면 중앙에 올 때
        end: "bottom center",         // 섹션의 하단이 화면 중앙에 올 때
        scrub: 1,                     // 스크롤 속도에 맞춰 부드럽게 추적 (숫자가 높을수록 부드러움)
        markers: true,                // 가이드 라인 표시 (개발용)
    },
    
    // 2. 변화시킬 속성들
    x: 500,               // 오른쪽으로 이동
    rotation: 720,        // 두 바퀴 회전
    scale: 1.5,           // 크기 확대
    borderRadius: "50%",  // 원으로 변함
    backgroundColor: "#2ed573" // 색상 변경
});