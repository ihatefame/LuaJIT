// LJX — lexer.
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "ljx/fe/Lexer.hpp"
#include "ljx/gc/GarbageCollector.hpp"
#include "ljx/rt/StringInterner.hpp"
#include "ljx/vm/Interpreter.hpp"

namespace ljx::fe {

namespace {
constexpr const char* kKeywords[] = {
    "and", "break", "do", "else", "elseif", "end", "false", "for", "function",
    "goto", "if", "in", "local", "nil", "not", "or", "repeat", "return",
    "then", "true", "until", "while",
};
constexpr std::size_t kKeywordCount = sizeof(kKeywords) / sizeof(kKeywords[0]);

bool IsDigit(int nChar) { return nChar >= '0' && nChar <= '9'; }
bool IsHexDigit(int nChar) {
    return IsDigit(nChar) || (nChar >= 'a' && nChar <= 'f') || (nChar >= 'A' && nChar <= 'F');
}
bool IsAlpha(int nChar) {
    return (nChar >= 'a' && nChar <= 'z') || (nChar >= 'A' && nChar <= 'Z') || nChar == '_';
}
}  // namespace

C_Lexer::C_Lexer(vm::C_Universe& uni, Reader_f fnReader, void* pReaderData,
                 vm::C_GcString* pChunkName) noexcept
    : m_pChunkName(pChunkName), m_fnReader(fnReader), m_pReaderData(pReaderData),
      m_pUniverse(&uni) {
    // Keyword recognition rides on interning: reserved byte = keyword index+1.
    for (std::size_t uI = 0; uI < kKeywordCount; ++uI) {
        vm::C_GcString* pStr = uni.Interner().Intern(kKeywords[uI]);
        pStr->m_Header.uExtra1 = static_cast<std::uint8_t>(uI + 1);
        uni.Gc().FixObject(&pStr->m_Header);
    }
    m_nChar = NextChar();
    Next();  // prime the current token
}

int C_Lexer::ReadMore() noexcept {
    std::size_t uSize = 0;
    const char* pBlock = m_fnReader ? m_fnReader(m_pReaderData, &uSize) : nullptr;
    if (!pBlock || uSize == 0) return -1;
    m_pCursor = pBlock;
    m_pWindowEnd = pBlock + uSize;
    return static_cast<unsigned char>(*m_pCursor++);
}

int C_Lexer::NextChar() noexcept {
    return m_pCursor < m_pWindowEnd ? static_cast<unsigned char>(*m_pCursor++) : ReadMore();
}

char C_Lexer::PeekChar() const noexcept {
    return m_pCursor < m_pWindowEnd ? *m_pCursor : '\0';
}

[[noreturn]] void C_Lexer::ErrorAt(core::BcLine_t uLine, const char* sMessage, ...) {
    char vBuffer[256];
    va_list args;
    va_start(args, sMessage);
    std::vsnprintf(vBuffer, sizeof vBuffer, sMessage, args);
    va_end(args);
    vm::RaiseError(*m_pUniverse, "%s:%u: %s",
                   m_pChunkName ? m_pChunkName->Data() : "?", uLine, vBuffer);
}

void C_Lexer::Next() {
    m_uLastLine = m_uLine;
    if (m_bHasLookahead) {
        m_tokCurrent = m_tokLookahead;
        m_bHasLookahead = false;
        return;
    }
    m_tokCurrent.eKind = Scan(m_tokCurrent);
}

const Token_t& C_Lexer::Lookahead() {
    if (!m_bHasLookahead) {
        m_tokLookahead.eKind = Scan(m_tokLookahead);
        m_bHasLookahead = true;
    }
    return m_tokLookahead;
}

// Long-bracket detection: at '[', counts '='s; returns level or -1.
int C_Lexer::CheckLongBracket() noexcept {
    int nLevel = 0;
    int nChar = m_nChar;  // == '['
    (void)nChar;
    const char* pSave = m_pCursor;
    const char* pSaveEnd = m_pWindowEnd;
    int nNext = NextChar();
    while (nNext == '=') {
        ++nLevel;
        nNext = NextChar();
    }
    if (nNext == '[') {
        m_nChar = NextChar();
        return nLevel;
    }
    // Not a long bracket: rewind (single-buffer readers only refill once, so
    // the window pointers are still valid).
    m_pCursor = pSave;
    m_pWindowEnd = pSaveEnd;
    return -1;
}

void C_Lexer::ReadLongString(Token_t& tokOut, int nLevel, bool bIsComment) {
    m_sBuffer.clear();
    if (m_nChar == '\n' || m_nChar == '\r') {  // skip first newline
        ++m_uLine;
        m_nChar = NextChar();
    }
    for (;;) {
        if (m_nChar < 0) ErrorAt(m_uLine, "unfinished long %s", bIsComment ? "comment" : "string");
        if (m_nChar == ']') {
            int nCount = 0;
            int nNext = NextChar();
            while (nNext == '=') {
                ++nCount;
                nNext = NextChar();
            }
            if (nCount == nLevel && nNext == ']') {
                m_nChar = NextChar();
                if (!bIsComment)
                    tokOut.tvValue = vm::TValue_t::GcObject(
                        vm::EValueTag::String, m_pUniverse->Interner().Intern(m_sBuffer));
                return;
            }
            m_sBuffer.push_back(']');
            for (int nI = 0; nI < nCount; ++nI) m_sBuffer.push_back('=');
            m_nChar = nNext;
            continue;
        }
        if (m_nChar == '\n') ++m_uLine;
        m_sBuffer.push_back(static_cast<char>(m_nChar));
        m_nChar = NextChar();
    }
}

void C_Lexer::ReadString(Token_t& tokOut, int nQuote) {
    m_sBuffer.clear();
    m_nChar = NextChar();
    while (m_nChar != nQuote) {
        if (m_nChar < 0 || m_nChar == '\n') ErrorAt(m_uLine, "unfinished string");
        if (m_nChar == '\\') {
            m_nChar = NextChar();
            switch (m_nChar) {
                case 'n': m_sBuffer.push_back('\n'); break;
                case 't': m_sBuffer.push_back('\t'); break;
                case 'r': m_sBuffer.push_back('\r'); break;
                case 'a': m_sBuffer.push_back('\a'); break;
                case 'b': m_sBuffer.push_back('\b'); break;
                case 'f': m_sBuffer.push_back('\f'); break;
                case 'v': m_sBuffer.push_back('\v'); break;
                case '\\': m_sBuffer.push_back('\\'); break;
                case '"': m_sBuffer.push_back('"'); break;
                case '\'': m_sBuffer.push_back('\''); break;
                case '\n': m_sBuffer.push_back('\n'); ++m_uLine; break;
                case 'x': {
                    int nValue = 0;
                    for (int nI = 0; nI < 2; ++nI) {
                        m_nChar = NextChar();
                        if (!IsHexDigit(m_nChar)) ErrorAt(m_uLine, "hexadecimal digit expected");
                        nValue = nValue * 16 +
                                 (IsDigit(m_nChar) ? m_nChar - '0' : (m_nChar | 0x20) - 'a' + 10);
                    }
                    m_sBuffer.push_back(static_cast<char>(nValue));
                    break;
                }
                default: {
                    if (!IsDigit(m_nChar)) ErrorAt(m_uLine, "invalid escape sequence");
                    int nValue = 0;
                    for (int nI = 0; nI < 3 && IsDigit(m_nChar); ++nI) {
                        nValue = nValue * 10 + (m_nChar - '0');
                        m_nChar = NextChar();
                    }
                    if (nValue > 255) ErrorAt(m_uLine, "decimal escape too large");
                    m_sBuffer.push_back(static_cast<char>(nValue));
                    continue;  // m_nChar already advanced
                }
            }
            m_nChar = NextChar();
            continue;
        }
        m_sBuffer.push_back(static_cast<char>(m_nChar));
        m_nChar = NextChar();
    }
    m_nChar = NextChar();  // skip closing quote
    tokOut.tvValue = vm::TValue_t::GcObject(vm::EValueTag::String,
                                            m_pUniverse->Interner().Intern(m_sBuffer));
}

void C_Lexer::ReadNumber(Token_t& tokOut) {
    m_sBuffer.clear();
    const bool bHex = m_nChar == '0' && (PeekChar() == 'x' || PeekChar() == 'X');
    for (;;) {
        m_sBuffer.push_back(static_cast<char>(m_nChar));
        m_nChar = NextChar();
        const bool bExpChar = bHex ? (m_nChar == 'p' || m_nChar == 'P')
                                   : (m_nChar == 'e' || m_nChar == 'E');
        if (bExpChar) {
            m_sBuffer.push_back(static_cast<char>(m_nChar));
            m_nChar = NextChar();
            if (m_nChar == '+' || m_nChar == '-') continue;  // sign appended next loop
        }
        if (!(IsDigit(m_nChar) || m_nChar == '.' ||
              (bHex && (IsHexDigit(m_nChar) || m_nChar == 'x' || m_nChar == 'X'))))
            break;
    }
    char* pEnd = nullptr;
    const double flValue = std::strtod(m_sBuffer.c_str(), &pEnd);
    if (pEnd != m_sBuffer.c_str() + m_sBuffer.size())
        ErrorAt(m_uLine, "malformed number near '%s'", m_sBuffer.c_str());
    tokOut.tvValue = vm::TValue_t::Number(flValue);
}

ETokenKind C_Lexer::Scan(Token_t& tokOut) {
    for (;;) {
        if (IsAlpha(m_nChar)) {
            m_sBuffer.clear();
            do {
                m_sBuffer.push_back(static_cast<char>(m_nChar));
                m_nChar = NextChar();
            } while (IsAlpha(m_nChar) || IsDigit(m_nChar));
            vm::C_GcString* pName = m_pUniverse->Interner().Intern(m_sBuffer);
            tokOut.tvValue = vm::TValue_t::GcObject(vm::EValueTag::String, pName);
            if (pName->m_Header.uExtra1 > 0)
                return static_cast<ETokenKind>(
                    static_cast<std::uint16_t>(ETokenKind::And) + pName->m_Header.uExtra1 - 1);
            return ETokenKind::Name;
        }
        if (IsDigit(m_nChar) ||
            (m_nChar == '.' && IsDigit(static_cast<unsigned char>(PeekChar())))) {
            ReadNumber(tokOut);
            return ETokenKind::Number;
        }
        switch (m_nChar) {
            case -1: return ETokenKind::Eof;
            case '\n': ++m_uLine; [[fallthrough]];
            case ' ': case '\t': case '\r': m_nChar = NextChar(); continue;
            case '-': {
                m_nChar = NextChar();
                if (m_nChar != '-') return static_cast<ETokenKind>('-');
                m_nChar = NextChar();
                if (m_nChar == '[') {
                    const int nLevel = CheckLongBracket();
                    if (nLevel >= 0) {
                        Token_t tokDummy;
                        ReadLongString(tokDummy, nLevel, true);
                        continue;
                    }
                }
                while (m_nChar >= 0 && m_nChar != '\n') m_nChar = NextChar();
                continue;
            }
            case '[': {
                const int nLevel = CheckLongBracket();
                if (nLevel >= 0) {
                    ReadLongString(tokOut, nLevel, false);
                    return ETokenKind::String;
                }
                m_nChar = NextChar();
                return static_cast<ETokenKind>('[');
            }
            case '"': case '\'': {
                const int nQuote = m_nChar;
                ReadString(tokOut, nQuote);
                return ETokenKind::String;
            }
            case '=':
                m_nChar = NextChar();
                if (m_nChar == '=') { m_nChar = NextChar(); return ETokenKind::Eq; }
                return static_cast<ETokenKind>('=');
            case '~':
                m_nChar = NextChar();
                if (m_nChar == '=') { m_nChar = NextChar(); return ETokenKind::Ne; }
                return static_cast<ETokenKind>('~');
            case '<':
                m_nChar = NextChar();
                if (m_nChar == '=') { m_nChar = NextChar(); return ETokenKind::Le; }
                return static_cast<ETokenKind>('<');
            case '>':
                m_nChar = NextChar();
                if (m_nChar == '=') { m_nChar = NextChar(); return ETokenKind::Ge; }
                return static_cast<ETokenKind>('>');
            case '.': {
                m_nChar = NextChar();
                if (m_nChar == '.') {
                    m_nChar = NextChar();
                    if (m_nChar == '.') {
                        m_nChar = NextChar();
                        return ETokenKind::Ellipsis;
                    }
                    return ETokenKind::Concat;
                }
                return static_cast<ETokenKind>('.');
            }
            default: {
                const int nSingle = m_nChar;
                m_nChar = NextChar();
                return static_cast<ETokenKind>(nSingle);
            }
        }
    }
}

}  // namespace ljx::fe
