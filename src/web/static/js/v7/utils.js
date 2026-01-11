// Shared utilities for v7 UI.
(function() {
const v7 = window.v7 || (window.v7 = { api: {}, utils: {}, state: {}, render: {}, actions: {}, viewer: {}, products: {}, tabs: {}, copy: {} });

v7.utils.byId = (id) => document.getElementById(id);
window.byId = v7.utils.byId;
v7.utils.toPretty = (obj) => JSON.stringify(obj, null, 2);

v7.utils.safeText = (id, text) => {
    const el = v7.utils.byId(id);
    if (el) el.textContent = text;
};

v7.utils.clamp = (value, min, max) => Math.min(Math.max(value, min), max);

v7.utils.normalizeLab = (lab) => {
    let [l, a, b] = lab;
    const needsNormalize = l > 100 || a > 128 || b > 128 || a < -128 || b < -128;
    if (needsNormalize) {
        l = (l / 255) * 100;
        a = a - 128;
        b = b - 128;
    }
    return [l, a, b];
};

v7.utils.labToRgb = (lab) => {
    if (!Array.isArray(lab) || lab.length < 3) return null;
    let [l, a, b] = v7.utils.normalizeLab(lab);
    let y = (l + 16) / 116;
    let x = y + a / 500;
    let z = y - b / 200;

    const xyz = [x, y, z].map((v) => {
        const v3 = v ** 3;
        return v3 > 0.008856 ? v3 : (v - 16 / 116) / 7.787;
    });

    x = xyz[0] * 95.047;
    y = xyz[1] * 100.0;
    z = xyz[2] * 108.883;

    x /= 100;
    y /= 100;
    z /= 100;

    let r = x * 3.2406 + y * -1.5372 + z * -0.4986;
    let g = x * -0.9689 + y * 1.8758 + z * 0.0415;
    let b2 = x * 0.0557 + y * -0.2040 + z * 1.0570;

    const rgb = [r, g, b2].map((v) => {
        const vv = v <= 0.0031308 ? 12.92 * v : 1.055 * Math.pow(v, 1 / 2.4) - 0.055;
        return Math.round(v7.utils.clamp(vv, 0, 1) * 255);
    });
    return rgb;
};

v7.utils.rgbToHex = (rgb) => {
    if (!Array.isArray(rgb)) return null;
    const hex = rgb.map((v) => v.toString(16).padStart(2, "0")).join("");
    return `#${hex}`.toUpperCase();
};

v7.utils.labToHex = (lab) => {
    const rgb = v7.utils.labToRgb(lab);
    return rgb ? v7.utils.rgbToHex(rgb) : null;
};

v7.utils.REASON_MAP = {
    PATTERN_BASELINE_NOT_FOUND: {
        title: "패턴 기준 없음",
        desc: "패턴 기준이 없어 비교를 수행하지 못했습니다.",
        fix: "STD 등록 후 다시 검사하세요."
    },
    DELTAE_P95_HIGH: {
        title: "패턴 색차 과다",
        desc: "ΔE p95가 기준을 초과했습니다.",
        fix: "조명/정렬을 확인하세요."
    },
    DELTAE_MEAN_HIGH: {
        title: "패턴 평균 색차 과다",
        desc: "ΔE 평균이 기준을 초과했습니다.",
        fix: "조명 조건을 확인하세요."
    },
    SIGNATURE_CORR_LOW: {
        title: "패턴 상관 낮음",
        desc: "패턴 유사도가 기준 이하입니다.",
        fix: "렌즈 정렬/촬영 품질을 확인하세요."
    },
    COLOR_SHIFT_BLUE: {
        title: "색상 청색 이동",
        desc: "색상이 파란 방향으로 이동했습니다.",
        fix: "화이트밸런스/조명을 점검하세요."
    },
    COLOR_SHIFT_DARK: {
        title: "색상 어두움",
        desc: "밝기가 기준보다 낮습니다.",
        fix: "노출/조명을 점검하세요."
    },
    PATTERN_DOT_COVERAGE_LOW: {
        title: "도트 커버리지 낮음",
        desc: "도트 면적이 기준보다 낮습니다.",
        fix: "노출/포커스를 점검하세요."
    },
    PATTERN_DOT_COVERAGE_HIGH: {
        title: "도트 커버리지 높음",
        desc: "도트 면적이 기준보다 높습니다.",
        fix: "노출/포커스를 점검하세요."
    },
    PATTERN_EDGE_SHARPNESS_LOW: {
        title: "에지 선명도 낮음",
        desc: "도트 가장자리 선명도가 기준 이하입니다.",
        fix: "초점/노출을 점검하세요."
    },
    CENTER_NOT_IN_FRAME: {
        title: "중심 이탈",
        desc: "렌즈 중심이 프레임 기준을 벗어났습니다.",
        fix: "센터 정렬 후 재촬영하세요."
    },
    BLUR_LOW: {
        title: "블러",
        desc: "블러 지표가 기준 이하입니다.",
        fix: "초점/손떨림을 점검하세요."
    },
    ILLUMINATION_FAIL: {
        title: "조명 불균형",
        desc: "조명 대칭/균일도가 기준을 벗어났습니다.",
        fix: "조명 상태를 점검하세요."
    },
    CFG_MISMATCH: {
        title: "설정 불일치",
        desc: "활성 STD와 설정 해시가 다릅니다.",
        fix: "STD 재등록 또는 설정을 확인하세요."
    },
    INK_COUNT_MISMATCH_SUSPECTED: {
        title: "잉크 수 불확실",
        desc: "잉크 개수 추정이 불안정합니다.",
        fix: "샘플링/잉크수 설정을 점검하세요."
    }
};

v7.utils.getReasonInfo = (code) => {
    if (!code) {
        return { title: "사유 없음", desc: "사유 코드가 없습니다.", fix: "" };
    }
    return v7.utils.REASON_MAP[code] || {
        title: "알 수 없음",
        desc: code,
        fix: "로그를 확인하세요."
    };
};

})();
