// LJX — lexer.
// Layer 4.
//
// Zero-allocation scanning: the input window is a pair of raw pointers with a
// cold refill path; token strings intern directly into the current function's
// constant table (which simultaneously roots them against GC, dedupes, and
// pre-populates constant lookup). Keyword recognition rides on interning: the
// interned string's reserved byte (header hole) IS the keyword id — no lookup
// structure exists. Exactly one token of lookahead.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

#include "ljx/vm/Object.hpp"

namespace ljx::fe {

enum class ETokenKind : std::uint16_t {
    // Single-char tokens are their own ASCII codes; named tokens from 256 up.
    OfsNamed = 256,
    // keywords (ids match the interned-string reserved byte, order frozen)
    And, Break, Do, Else, ElseIf, End, False, For, Function, Goto, If, In,
    Local, Nil, Not, Or, Repeat, Return, Then, True, Until, While,
    // fork extensions: soft keywords resolved contextually
    Const, Continue,
    // multi-char operators
    Concat, Ellipsis, Eq, Ne, Le, Ge, ShL, ShR, AShR, SafeNav /* ?. */,
    Coalesce /* ?? */, Arrow /* -> */, CompoundAssign /* += -= … */,
    // literals & terminals
    Number, Name, String, Eof,
};

struct Token_t {
    ETokenKind eKind = ETokenKind::Eof;
    vm::TValue_t tvValue;   // number value or interned-string reference
};

// Streaming source supplier (files, memory, custom readers).
using Reader_f = const char* (*)(void* pUserData, std::size_t* puSize);

class C_Lexer {
public:
    C_Lexer(vm::C_Universe& uni, Reader_f fnReader, void* pReaderData,
            vm::C_GcString* pChunkName) noexcept;

    void Next();                              // advance current ← lookahead
    [[nodiscard]] const Token_t& Current() const noexcept { return m_tokCurrent; }
    [[nodiscard]] const Token_t& Lookahead();  // scan on demand
    [[nodiscard]] core::BcLine_t Line() const noexcept { return m_uLine; }

    // Peek at the raw next character (compound-assignment disambiguation).
    [[nodiscard]] char PeekChar() const noexcept;

    [[noreturn]] void ErrorAt(core::BcLine_t uLine, const char* sMessage, ...);

public:
    [[nodiscard]] vm::C_Universe& Universe() noexcept { return *m_pUniverse; }
    [[nodiscard]] vm::C_GcString* ChunkName() const noexcept { return m_pChunkName; }
    [[nodiscard]] core::BcLine_t LastLine() const noexcept { return m_uLastLine; }

private:
    [[nodiscard]] int ReadMore() noexcept;    // cold refill path
    [[nodiscard]] int NextChar() noexcept;
    [[nodiscard]] ETokenKind Scan(Token_t& tokOut);
    void ReadString(Token_t& tokOut, int nQuote);
    void ReadLongString(Token_t& tokOut, int nLevel, bool bIsComment);
    void ReadNumber(Token_t& tokOut);
    [[nodiscard]] int CheckLongBracket() noexcept;  // returns level or -1
    int m_nChar = 0;                    // current character (-1 = EOF)
    std::string m_sBuffer;              // token text scratch

    const char* m_pCursor = nullptr;   // hot window: two pointer compares/char
    const char* m_pWindowEnd = nullptr;
    Token_t m_tokCurrent;
    Token_t m_tokLookahead;             // eKind == Eof means "none buffered"
    bool m_bHasLookahead = false;
    core::BcLine_t m_uLine = 1;
    core::BcLine_t m_uLastLine = 1;
    vm::C_GcString* m_pChunkName = nullptr;
    Reader_f m_fnReader = nullptr;
    void* m_pReaderData = nullptr;
    vm::C_Universe* m_pUniverse = nullptr;
};

}  // namespace ljx::fe
