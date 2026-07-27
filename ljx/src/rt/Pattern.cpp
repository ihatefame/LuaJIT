// LJX — Lua 5.1 pattern matching engine (see include/ljx/rt/Pattern.hpp).
//
// A faithful port of the reference algorithm: recursive backtracking with
// greedy (`*`, `+`, `-` lazy) quantifiers, character classes, sets, capture
// stack, `%b` balance and `%f` frontier. The recursion guard replaces the
// C-stack overflow of pathological patterns with a Lua error.

#include "ljx/rt/Pattern.hpp"

#include <cctype>
#include <cstring>

#include "ljx/vm/Interpreter.hpp"

namespace ljx::rt {

using vm::RaiseError;

namespace {
inline constexpr std::uint32_t kMaxMatchDepth = 200;
inline constexpr char kEscape = '%';
}  // namespace

bool C_PatternMatcher::MatchClass(unsigned char uC, unsigned char uClass) const noexcept {
    bool bRes;
    switch (std::tolower(uClass)) {
        case 'a': bRes = std::isalpha(uC); break;
        case 'c': bRes = std::iscntrl(uC); break;
        case 'd': bRes = std::isdigit(uC); break;
        case 'l': bRes = std::islower(uC); break;
        case 'p': bRes = std::ispunct(uC); break;
        case 's': bRes = std::isspace(uC); break;
        case 'u': bRes = std::isupper(uC); break;
        case 'w': bRes = std::isalnum(uC); break;
        case 'x': bRes = std::isxdigit(uC); break;
        default: return uClass == uC;
    }
    if (std::isupper(uClass)) bRes = !bRes;
    return bRes;
}

bool C_PatternMatcher::MatchBracket(unsigned char uC, const char* pP,
                                    const char* pEc) const noexcept {
    bool bWanted = true;
    if (pP[1] == '^') {
        bWanted = false;
        ++pP;
    }
    while (++pP < pEc) {
        if (*pP == kEscape) {
            ++pP;
            if (MatchClass(uC, static_cast<unsigned char>(*pP))) return bWanted;
        } else if (pP[1] == '-' && pP + 2 < pEc) {
            if (static_cast<unsigned char>(pP[0]) <= uC &&
                uC <= static_cast<unsigned char>(pP[2]))
                return bWanted;
            pP += 2;
        } else if (static_cast<unsigned char>(*pP) == uC) {
            return bWanted;
        }
    }
    return !bWanted;
}

bool C_PatternMatcher::SingleMatch(const char* pS, const char* pP,
                                   const char* pEp) const noexcept {
    if (pS >= m_pSrcEnd) return false;
    const auto uC = static_cast<unsigned char>(*pS);
    switch (*pP) {
        case '.': return true;
        case kEscape: return MatchClass(uC, static_cast<unsigned char>(pP[1]));
        case '[': return MatchBracket(uC, pP, pEp - 1);
        default: return static_cast<unsigned char>(*pP) == uC;
    }
}

// One past the end of the single-item class starting at pP.
const char* C_PatternMatcher::ClassEnd(const char* pP) const {
    switch (*pP++) {
        case kEscape: {
            if (pP >= m_pPatEnd)
                RaiseError(*m_pUniverse, "malformed pattern (ends with '%%')");
            return pP + 1;
        }
        case '[': {
            if (pP < m_pPatEnd && *pP == '^') ++pP;
            do {  // look for a ']'; the FIRST position may hold a literal one
                if (pP >= m_pPatEnd)
                    RaiseError(*m_pUniverse, "malformed pattern (missing ']')");
                if (*pP++ == kEscape && pP < m_pPatEnd) ++pP;
            } while (pP >= m_pPatEnd || *pP != ']');
            return pP + 1;
        }
        default:
            return pP;
    }
}

const char* C_PatternMatcher::MatchBalance(const char* pS, const char* pP) const {
    if (pP + 1 >= m_pPatEnd)
        RaiseError(*m_pUniverse, "missing arguments to '%%b'");
    if (pS >= m_pSrcEnd || *pS != pP[0]) return nullptr;
    const char cBegin = pP[0];
    const char cEnd = pP[1];
    std::uint32_t uCount = 1;
    const char* pC = pS;
    while (++pC < m_pSrcEnd) {
        if (*pC == cEnd) {
            if (--uCount == 0) return pC + 1;
        } else if (*pC == cBegin) {
            ++uCount;
        }
    }
    return nullptr;
}

const char* C_PatternMatcher::MaxExpand(const char* pS, const char* pP, const char* pEp) {
    std::ptrdiff_t nCount = 0;
    while (SingleMatch(pS + nCount, pP, pEp)) ++nCount;
    // Longest first, backing off one repetition per failed tail.
    while (nCount >= 0) {
        if (const char* pRes = Match(pS + nCount, pEp + 1)) return pRes;
        --nCount;
    }
    return nullptr;
}

const char* C_PatternMatcher::MinExpand(const char* pS, const char* pP, const char* pEp) {
    for (;;) {
        if (const char* pRes = Match(pS, pEp + 1)) return pRes;
        if (SingleMatch(pS, pP, pEp))
            ++pS;
        else
            return nullptr;
    }
}

const char* C_PatternMatcher::StartCapture(const char* pS, const char* pP,
                                           std::ptrdiff_t nWhat) {
    if (m_uLevel >= kMaxCaptures) RaiseError(*m_pUniverse, "too many captures");
    m_vCapture[m_uLevel].pInit = pS;
    m_vCapture[m_uLevel].nLen = nWhat;
    ++m_uLevel;
    const char* pRes = Match(pS, pP);
    if (!pRes) --m_uLevel;   // undo on failure
    return pRes;
}

const char* C_PatternMatcher::EndCapture(const char* pS, const char* pP) {
    std::int32_t nL = -1;
    for (std::int32_t nI = static_cast<std::int32_t>(m_uLevel) - 1; nI >= 0; --nI)
        if (m_vCapture[nI].nLen == kCapUnfinished) {
            nL = nI;
            break;
        }
    if (nL < 0) RaiseError(*m_pUniverse, "invalid pattern capture");
    m_vCapture[nL].nLen = pS - m_vCapture[nL].pInit;
    const char* pRes = Match(pS, pP);
    if (!pRes) m_vCapture[nL].nLen = kCapUnfinished;   // undo on failure
    return pRes;
}

std::uint32_t C_PatternMatcher::CheckCapture(char cIdx) const {
    const auto nIdx = static_cast<std::int32_t>(cIdx - '1');
    if (nIdx < 0 || nIdx >= static_cast<std::int32_t>(m_uLevel) ||
        m_vCapture[nIdx].nLen == kCapUnfinished)
        RaiseError(*m_pUniverse, "invalid capture index %%%d", nIdx + 1);
    return static_cast<std::uint32_t>(nIdx);
}

const char* C_PatternMatcher::MatchCaptureRef(const char* pS, std::uint32_t uIdx) {
    const std::ptrdiff_t nLen = m_vCapture[uIdx].nLen;
    if (m_pSrcEnd - pS >= nLen &&
        std::memcmp(m_vCapture[uIdx].pInit, pS, static_cast<std::size_t>(nLen)) == 0)
        return pS + nLen;
    return nullptr;
}

const char* C_PatternMatcher::Match(const char* pS, const char* pP) {
    if (++m_uDepth > kMaxMatchDepth) {
        --m_uDepth;
        RaiseError(*m_pUniverse, "pattern too complex");
    }
    const char* pRes;
    while (pP < m_pPatEnd) {
        switch (*pP) {
            case '(': {
                pRes = pP[1] == ')' ? StartCapture(pS, pP + 2, kCapPosition)
                                    : StartCapture(pS, pP + 1, kCapUnfinished);
                --m_uDepth;
                return pRes;
            }
            case ')': {
                pRes = EndCapture(pS, pP + 1);
                --m_uDepth;
                return pRes;
            }
            case '$': {
                if (pP + 1 == m_pPatEnd) {
                    --m_uDepth;
                    return pS == m_pSrcEnd ? pS : nullptr;
                }
                break;   // a '$' elsewhere is a literal
            }
            case kEscape: {
                switch (pP[1]) {
                    case 'b': {
                        const char* pB = MatchBalance(pS, pP + 2);
                        if (!pB) {
                            --m_uDepth;
                            return nullptr;
                        }
                        pS = pB;
                        pP += 4;
                        continue;
                    }
                    case 'f': {
                        pP += 2;
                        if (pP >= m_pPatEnd || *pP != '[')
                            RaiseError(*m_pUniverse, "missing '[' after '%%f'");
                        const char* pEp = ClassEnd(pP);
                        const auto uPrev = static_cast<unsigned char>(
                            pS == m_pSrcInit ? '\0' : pS[-1]);
                        const auto uCur =
                            static_cast<unsigned char>(pS < m_pSrcEnd ? *pS : '\0');
                        if (!MatchBracket(uPrev, pP, pEp - 1) &&
                            MatchBracket(uCur, pP, pEp - 1)) {
                            pP = pEp;
                            continue;
                        }
                        --m_uDepth;
                        return nullptr;
                    }
                    default: {
                        if (std::isdigit(static_cast<unsigned char>(pP[1]))) {
                            const char* pB = MatchCaptureRef(pS, CheckCapture(pP[1]));
                            if (!pB) {
                                --m_uDepth;
                                return nullptr;
                            }
                            pS = pB;
                            pP += 2;
                            continue;
                        }
                        break;
                    }
                }
                break;
            }
            default:
                break;
        }
        // Default: a single-item class, possibly quantified.
        const char* pEp = ClassEnd(pP);
        const bool bMatches = SingleMatch(pS, pP, pEp);
        const char cSuffix = pEp < m_pPatEnd ? *pEp : '\0';
        switch (cSuffix) {
            case '?': {
                if (bMatches) {
                    if ((pRes = Match(pS + 1, pEp + 1))) {
                        --m_uDepth;
                        return pRes;
                    }
                }
                pP = pEp + 1;
                continue;
            }
            case '+': {
                pRes = bMatches ? MaxExpand(pS + 1, pP, pEp) : nullptr;
                --m_uDepth;
                return pRes;
            }
            case '*': {
                pRes = MaxExpand(pS, pP, pEp);
                --m_uDepth;
                return pRes;
            }
            case '-': {
                pRes = MinExpand(pS, pP, pEp);
                --m_uDepth;
                return pRes;
            }
            default: {
                if (!bMatches) {
                    --m_uDepth;
                    return nullptr;
                }
                ++pS;
                pP = pEp;
                continue;
            }
        }
    }
    --m_uDepth;
    return pS;
}

}  // namespace ljx::rt
