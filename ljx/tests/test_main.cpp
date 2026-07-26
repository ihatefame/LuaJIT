// LJX — C++ unit tests for the core data structures.
#include <cstdio>
#include <cstring>

#include "ljx/gc/GarbageCollector.hpp"
#include "ljx/rt/StringInterner.hpp"
#include "ljx/vm/Interpreter.hpp"
#include "ljx/vm/Object.hpp"

using namespace ljx;

static int g_nFailures = 0;
#define CHECK(cond)                                                          \
    do {                                                                     \
        if (!(cond)) {                                                       \
            std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);      \
            ++g_nFailures;                                                   \
        }                                                                    \
    } while (0)

static void TestValues() {
    using vm::TValue_t;
    CHECK(TValue_t::Nil().IsNil());
    CHECK(!TValue_t::Nil().IsTruthy());
    CHECK(!TValue_t::Boolean(false).IsTruthy());
    CHECK(TValue_t::Boolean(true).IsTruthy());
    CHECK(TValue_t::Number(0.0).IsTruthy());   // 0 is truthy in Lua
    CHECK(TValue_t::Number(3.5).IsDouble());
    CHECK(TValue_t::Number(3.5).AsDouble() == 3.5);
    CHECK(TValue_t::Number(-1e300).IsNumber());
    CHECK(TValue_t::Integer(-7).IsInteger());
    CHECK(TValue_t::Integer(-7).AsInteger() == -7);
    CHECK(core::NumToBit(4294967296.0 + 5.0) == 5);
    std::int32_t nOut = 0;
    CHECK(core::NumToInt32Check(42.0, nOut) && nOut == 42);
    CHECK(!core::NumToInt32Check(42.5, nOut));
    CHECK(!core::NumToInt32Check(0.0 / 0.0, nOut));
}

static void TestInterner(vm::C_Universe& uni) {
    vm::C_GcString* pA = uni.Interner().Intern("hello");
    vm::C_GcString* pB = uni.Interner().Intern("hello");
    vm::C_GcString* pC = uni.Interner().Intern("world");
    CHECK(pA == pB);                       // interning = pointer identity
    CHECK(pA != pC);
    CHECK(pA->Length() == 5);
    CHECK(std::memcmp(pA->Data(), "hello", 6) == 0);  // NUL-terminated
    // Force a resize.
    char vName[32];
    for (int nI = 0; nI < 1000; ++nI) {
        std::snprintf(vName, sizeof vName, "sym_%d", nI);
        (void)uni.Interner().Intern(vName);
    }
    CHECK(uni.Interner().Intern("sym_567") == uni.Interner().Intern("sym_567"));
}

static void TestTables(vm::C_Universe& uni) {
    using vm::TValue_t;
    vm::C_GcTable* pTab = vm::C_GcTable::New(uni, 0, 0);
    // Array-append growth.
    for (int nI = 1; nI <= 100; ++nI)
        *pTab->Set(uni, TValue_t::Number(nI)) = TValue_t::Number(nI * 10);
    CHECK(pTab->Length(uni) == 100);
    const TValue_t* pSlot = pTab->Get(uni, TValue_t::Number(57.0));
    CHECK(pSlot && pSlot->AsDouble() == 570.0);
    // String keys + collisions + Brent eviction.
    char vKey[32];
    for (int nI = 0; nI < 500; ++nI) {
        std::snprintf(vKey, sizeof vKey, "key_%d", nI);
        *pTab->Set(uni, TValue_t::GcObject(vm::EValueTag::String,
                                           uni.Interner().Intern(vKey))) =
            TValue_t::Number(nI);
    }
    for (int nI = 0; nI < 500; ++nI) {
        std::snprintf(vKey, sizeof vKey, "key_%d", nI);
        const TValue_t* pVal = pTab->GetStr(uni, uni.Interner().Intern(vKey));
        CHECK(pVal && pVal->AsDouble() == nI);
    }
    // Delete (nil store) keeps the node (dead-key stability).
    *pTab->Set(uni, TValue_t::GcObject(vm::EValueTag::String,
                                       uni.Interner().Intern("key_250"))) =
        TValue_t::Nil();
    const TValue_t* pDead = pTab->GetStr(uni, uni.Interner().Intern("key_250"));
    CHECK(pDead && pDead->IsNil());
    // -0 canonicalization.
    *pTab->Set(uni, TValue_t::Number(-0.0)) = TValue_t::Number(99);
    const TValue_t* pZero = pTab->Get(uni, TValue_t::Number(0.0));
    CHECK(pZero && pZero->AsDouble() == 99);
}

static void TestGc(vm::C_Universe& uni) {
    const auto uBefore = uni.Gc().Stats().uMajorCollections;
    // Allocate garbage tables, then collect; globals must survive.
    for (int nI = 0; nI < 10000; ++nI) (void)vm::C_GcTable::New(uni, 4, 0);
    uni.Gc().CollectNow();
    CHECK(uni.Gc().Stats().uMajorCollections == uBefore + 1);
    CHECK(uni.Interner().Intern("hello") != nullptr);
    vm::C_GcTable* pGlobals = uni.Globals();
    CHECK(pGlobals != nullptr);
}

int main() {
    TestValues();
    vm::C_Universe* pUni = vm::C_Universe::Create();
    if (!pUni) {
        std::printf("FAIL: cannot create universe\n");
        return 1;
    }
    vm::OpenStdLib(*pUni);
    TestInterner(*pUni);
    TestTables(*pUni);
    TestGc(*pUni);
    pUni->Destroy();
    if (g_nFailures) {
        std::printf("%d unit-test failure(s)\n", g_nFailures);
        return 1;
    }
    std::printf("all unit tests passed\n");
    return 0;
}
