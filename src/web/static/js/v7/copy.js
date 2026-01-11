// UI copy constants for v7 UI.
(function() {
const v7 = window.v7 || (window.v7 = { api: {}, utils: {}, state: {}, render: {}, actions: {}, viewer: {}, products: {}, tabs: {}, copy: {} });

v7.copy = {
    buttons: {
        inspectStart: "검사 실행",
        inspectRunning: "검사 중...",
        registerStart: "등록 및 검증 시작",
        registerRunning: "등록 중...",
        testStart: "테스트 실행",
        testRunning: "테스트 중..."
    },
    labels: {
        regSuccess: "SUCCESS",
        regFailed: "FAILED",
        regDone: "검증 완료",
        regFail: "검증 실패",
        autoActivated: "AUTO ACTIVATED",
        activationFailed: "ACTIVATION FAILED",
        registered: "REGISTERED",
        detailOpen: "간략 보기",
        detailClose: "상세 보기",
        filePlaceholder: "파일을 드래그하거나 클릭해서 선택하세요."
    },
    alerts: {
        needProductName: "제품명을 입력하세요.",
        needInkCount: "잉크 개수를 입력하세요.",
        needInspectFile: "검사 이미지를 선택하세요.",
        needProductSelect: "제품을 선택하세요.",
        inspectFail: "검사 실패",
        registerFail: "등록 실패",
        statusFail: "상태 조회 실패",
        activateFail: "활성화 실패",
        autoActivateFail: "자동 활성화 실패",
        testFail: "테스트 실패",
        retakeChecklist: "재촬영 체크리스트를 확인하세요.",
        shareLinkCopied: "결과 공유 링크를 복사했습니다.",
        confirmed: "확인 완료 처리했습니다.",
        requestedApproval: "승인자에게 요청했습니다.",
        pinnedEvidence: "근거 패널을 고정했습니다.",
        notePrompt: "조치 메모를 남겨주세요."
    }
};

v7.t = (path, fallback = "") => {
    const parts = path.split(".");
    let node = v7.copy;
    for (const key of parts) {
        if (!node || typeof node !== "object" || !(key in node)) return fallback;
        node = node[key];
    }
    return typeof node === "string" ? node : fallback;
};

})();
