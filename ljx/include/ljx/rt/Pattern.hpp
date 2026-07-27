// LJX — Lua 5.1 string patterns: the classic backtracking engine.
//
// The matcher is deliberately independent of the stack protocol: it works on
// raw byte ranges and records captures as (pointer, length) pairs. The
// string-library entry points in Lib.cpp translate those into Lua values.
// Patterns are NOT regular expressions — no alternation, no grouping except
// captures — which is exactly what keeps this a few hundred lines with no
// compilation step.
#pragma once

#include <cstddef>
#include <cstdint>

namespace ljx::vm {
class C_Universe;
}

namespace ljx::rt {

inline constexpr std::uint32_t kMaxCaptures = 32;
inline constexpr std::ptrdiff_t kCapPosition = -2;    // position capture `()`
inline constexpr std::ptrdiff_t kCapUnfinished = -1;  // still open

struct MatchCapture_t {
    const char* pInit = nullptr;
    std::ptrdiff_t nLen = 0;
};

class C_PatternMatcher {
public:
    C_PatternMatcher(vm::C_Universe& uni, const char* pSrcInit, const char* pSrcEnd,
                     const char* pPatInit, const char* pPatEnd) noexcept
        : m_pUniverse(&uni),
          m_pSrcInit(pSrcInit),
          m_pSrcEnd(pSrcEnd),
          m_pPatEnd(pPatEnd) {
        (void)pPatInit;
    }

    // Matches the pattern (from pP) against the subject at exactly pS.
    // Returns one past the match end, or nullptr if there is no match here.
    // Captures accumulate in m_vCapture[0..m_uLevel).
    [[nodiscard]] const char* Match(const char* pS, const char* pP);

    std::uint32_t m_uLevel = 0;
    MatchCapture_t m_vCapture[kMaxCaptures];

private:
    [[nodiscard]] bool MatchClass(unsigned char uC, unsigned char uClass) const noexcept;
    [[nodiscard]] bool MatchBracket(unsigned char uC, const char* pP,
                                    const char* pEc) const noexcept;
    [[nodiscard]] bool SingleMatch(const char* pS, const char* pP,
                                   const char* pEp) const noexcept;
    [[nodiscard]] const char* ClassEnd(const char* pP) const;
    [[nodiscard]] const char* MatchBalance(const char* pS, const char* pP) const;
    [[nodiscard]] const char* MaxExpand(const char* pS, const char* pP, const char* pEp);
    [[nodiscard]] const char* MinExpand(const char* pS, const char* pP, const char* pEp);
    [[nodiscard]] const char* StartCapture(const char* pS, const char* pP,
                                           std::ptrdiff_t nWhat);
    [[nodiscard]] const char* EndCapture(const char* pS, const char* pP);
    [[nodiscard]] const char* MatchCaptureRef(const char* pS, std::uint32_t uIdx);
    [[nodiscard]] std::uint32_t CheckCapture(char cIdx) const;

    vm::C_Universe* m_pUniverse;
    const char* m_pSrcInit;
    const char* m_pSrcEnd;
    const char* m_pPatEnd;
    std::uint32_t m_uDepth = 0;   // backtracking recursion guard
};

}  // namespace ljx::rt
