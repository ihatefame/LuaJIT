// LJX — single-pass recursive-descent parser emitting registerized bytecode.
// Ported design: LuaJIT/Lua 5.1 expression-descriptor codegen (delayed
// discharge, jump lists threaded through Jmp D-fields, test-and-copy
// materialization) with the LJX opcode set.
#include <cstdio>
#include <cstring>
#include <unordered_map>
#include <vector>

#include "ljx/fe/Parser.hpp"
#include "ljx/gc/GarbageCollector.hpp"
#include "ljx/rt/StringInterner.hpp"
#include "ljx/vm/Interpreter.hpp"

namespace ljx::fe {

using vm::BcIns_t;
using vm::C_GcProto;
using vm::C_GcString;
using vm::EBcOp;
using vm::TValue_t;

namespace {

constexpr std::uint32_t kNoJump = kNoJumpPos;
constexpr std::uint8_t kNoRegSentinel = vm::kNoReg;

// Indexed-key encoding in ExpDesc_t::uAux.
constexpr std::uint32_t kKeyStrBase = 0x10000;   // uAux-kKeyStrBase = kgc index
constexpr std::uint32_t kKeyByteBase = 0x30000;  // uAux-kKeyByteBase = byte literal

struct UpvalDesc_t {
    C_GcString* pName;
    bool bFromParentLocal;
    std::uint8_t uIndex;
};

struct BlockScope_t {
    BlockScope_t* pPrev;
    std::uint32_t uFirstLocal;
    std::uint32_t posBreakList = kNoJump;
    bool bIsLoop;
};

struct FuncState_t {
    FuncState_t* pParent = nullptr;
    std::vector<BcIns_t> vCode;
    std::vector<core::BcLine_t> vLines;
    std::vector<double> vNum;
    std::unordered_map<std::uint64_t, std::uint32_t> mapNum;
    std::vector<TValue_t> vGc;  // strings + child protos, KStr/FNew index space
    std::unordered_map<std::uint64_t, std::uint32_t> mapGc;
    std::vector<C_GcString*> vLocals;  // active locals by slot
    std::vector<UpvalDesc_t> vUpvals;
    BlockScope_t* pBlock = nullptr;
    std::uint32_t uNumActive = 0;
    std::uint32_t uFreeReg = 0;
    std::uint32_t uFrameMax = 2;
    std::uint8_t uNumParams = 0;
    bool bAnyCaptured = false;
};

struct Ctx_t {
    C_Lexer& lex;
    vm::C_Universe& uni;
    FuncState_t* fs = nullptr;

    // ---- token helpers -----------------------------------------------------
    [[nodiscard]] ETokenKind Tok() const { return lex.Current().eKind; }
    void Next() { lex.Next(); }
    [[nodiscard]] bool Opt(ETokenKind eKind) {
        if (Tok() != eKind) return false;
        Next();
        return true;
    }
    void Expect(ETokenKind eKind, const char* sWhat) {
        if (Tok() != eKind) lex.ErrorAt(lex.Line(), "'%s' expected", sWhat);
        Next();
    }
    [[nodiscard]] C_GcString* ExpectName() {
        if (Tok() != ETokenKind::Name) lex.ErrorAt(lex.Line(), "<name> expected");
        auto* pName = static_cast<C_GcString*>(lex.Current().tvValue.AsGcPointer());
        Next();
        return pName;
    }
    [[noreturn]] void Error(const char* sMessage) { lex.ErrorAt(lex.Line(), "%s", sMessage); }

    // ---- emitter -----------------------------------------------------------
    [[nodiscard]] std::uint32_t Pos() const {
        return static_cast<std::uint32_t>(fs->vCode.size());
    }
    std::uint32_t Emit(BcIns_t ins) {
        fs->vCode.push_back(ins);
        fs->vLines.push_back(lex.LastLine());
        return Pos() - 1;
    }
    std::uint32_t EmitAD(EBcOp eOp, std::uint8_t uA, std::uint16_t uD) {
        return Emit(BcIns_t::MakeAD(eOp, uA, uD));
    }
    std::uint32_t EmitABC(EBcOp eOp, std::uint8_t uA, std::uint8_t uB, std::uint8_t uC) {
        return Emit(BcIns_t::MakeABC(eOp, uA, uB, uC));
    }

    // ---- registers ---------------------------------------------------------
    void ReserveRegs(std::uint32_t uCount) {
        fs->uFreeReg += uCount;
        if (fs->uFreeReg + 2 > fs->uFrameMax) {
            if (fs->uFreeReg + 2 > vm::kMaxFrameSlots) Error("function or expression too complex");
            fs->uFrameMax = fs->uFreeReg + 2;
        }
    }
    void FreeReg(std::uint32_t uReg) {
        if (uReg >= fs->uNumActive) {
            --fs->uFreeReg;
            if (uReg != fs->uFreeReg) Error("internal: register free order");
        }
    }
    void FreeExp(const ExpDesc_t& e) {
        if (e.eKind == EExpKind::NonReloc) FreeReg(e.payload.slotPair.uInfo);
    }

    // ---- constants ---------------------------------------------------------
    std::uint32_t NumConst(double flValue) {
        const std::uint64_t uBits = std::bit_cast<std::uint64_t>(flValue);
        auto it = fs->mapNum.find(uBits);
        if (it != fs->mapNum.end()) return it->second;
        const auto uIdx = static_cast<std::uint32_t>(fs->vNum.size());
        if (uIdx >= 0xffff) Error("too many number constants");
        fs->vNum.push_back(flValue);
        fs->mapNum.emplace(uBits, uIdx);
        return uIdx;
    }
    std::uint32_t GcConst(const TValue_t& tvValue) {
        auto it = fs->mapGc.find(tvValue.uRaw);
        if (it != fs->mapGc.end()) return it->second;
        const auto uIdx = static_cast<std::uint32_t>(fs->vGc.size());
        if (uIdx >= 0xffff) Error("too many constants");
        fs->vGc.push_back(tvValue);
        fs->mapGc.emplace(tvValue.uRaw, uIdx);
        return uIdx;
    }
    std::uint32_t StrConst(C_GcString* pStr) {
        return GcConst(TValue_t::GcObject(vm::EValueTag::String, pStr));
    }

    // ---- jump lists (threaded through Jmp D fields, absolute positions) ----
    [[nodiscard]] std::uint32_t JumpNext(std::uint32_t uPos) const {
        const std::uint16_t uD = fs->vCode[uPos].D();
        return uD == 0xffff ? kNoJump : uD;
    }
    void SetJumpNext(std::uint32_t uPos, std::uint32_t uNext) {
        fs->vCode[uPos].uRaw = (fs->vCode[uPos].uRaw & 0xffff) |
                               (std::uint32_t{uNext == kNoJump ? 0xffffu
                                                               : static_cast<std::uint16_t>(uNext)}
                                << 16);
    }
    std::uint32_t EmitJump() {
        const std::uint32_t uPos = EmitAD(EBcOp::Jmp, static_cast<std::uint8_t>(fs->uFreeReg),
                                          0xffff);
        return uPos;
    }
    void Concat(std::uint32_t& posList, std::uint32_t uPos) {
        if (uPos == kNoJump) return;
        if (posList == kNoJump) {
            posList = uPos;
            return;
        }
        std::uint32_t uWalk = posList;
        while (JumpNext(uWalk) != kNoJump) uWalk = JumpNext(uWalk);
        SetJumpNext(uWalk, uPos);
    }
    // The instruction controlling a jump (compare/test before it, or itself).
    [[nodiscard]] std::uint32_t JumpControlPos(std::uint32_t uPos) const {
        if (uPos >= 1) {
            const EBcOp eOp = fs->vCode[uPos - 1].Op();
            if (eOp >= EBcOp::IsLt && eOp <= EBcOp::IsF) return uPos - 1;
        }
        return uPos;
    }
    void FixJump(std::uint32_t uPos, std::uint32_t uTarget) {
        const std::int32_t nOffset =
            static_cast<std::int32_t>(uTarget) - static_cast<std::int32_t>(uPos) - 1;
        if (nOffset < -static_cast<std::int32_t>(vm::kJumpBias) ||
            nOffset >= static_cast<std::int32_t>(vm::kJumpBias))
            Error("control structure too long");
        fs->vCode[uPos].uRaw =
            (fs->vCode[uPos].uRaw & 0xffff) |
            (static_cast<std::uint32_t>(nOffset + static_cast<std::int32_t>(vm::kJumpBias))
             << 16);
    }
    bool PatchTestReg(std::uint32_t uJumpPos, std::uint8_t uReg) {
        const std::uint32_t uCtrl = JumpControlPos(uJumpPos);
        BcIns_t& ins = fs->vCode[uCtrl];
        const EBcOp eOp = ins.Op();
        if (eOp != EBcOp::IsTC && eOp != EBcOp::IsFC) return false;  // comparison
        if (uReg != kNoRegSentinel && uReg != static_cast<std::uint8_t>(ins.D())) {
            ins.uRaw = (ins.uRaw & ~0xff00u) | (std::uint32_t{uReg} << 8);
        } else {
            // Value unused (or already in place): degrade to the pure test.
            ins = BcIns_t::MakeAD(static_cast<EBcOp>(static_cast<std::uint8_t>(eOp) + 2), 0,
                                  ins.D());
        }
        return true;
    }
    [[nodiscard]] bool ListNeedsValue(std::uint32_t posList) const {
        for (std::uint32_t uWalk = posList; uWalk != kNoJump; uWalk = JumpNext(uWalk)) {
            const std::uint32_t uCtrl = JumpControlPos(uWalk);
            const EBcOp eOp = fs->vCode[uCtrl].Op();
            if (eOp != EBcOp::IsTC && eOp != EBcOp::IsFC) return true;
        }
        return false;
    }
    void PatchListAux(std::uint32_t posList, std::uint32_t uValueTarget, std::uint8_t uReg,
                      std::uint32_t uDefaultTarget) {
        std::uint32_t uWalk = posList;
        while (uWalk != kNoJump) {
            const std::uint32_t uNext = JumpNext(uWalk);
            if (PatchTestReg(uWalk, uReg))
                FixJump(uWalk, uValueTarget);
            else
                FixJump(uWalk, uDefaultTarget);
            uWalk = uNext;
        }
    }
    void PatchToHere(std::uint32_t posList) {
        PatchListAux(posList, Pos(), kNoRegSentinel, Pos());
    }
    void PatchTo(std::uint32_t posList, std::uint32_t uTarget) {
        PatchListAux(posList, uTarget, kNoRegSentinel, uTarget);
    }

    // ---- expression discharge ---------------------------------------------
    void DischargeVars(ExpDesc_t& e) {
        switch (e.eKind) {
            case EExpKind::Local:
                e.eKind = EExpKind::NonReloc;
                break;
            case EExpKind::Upvalue:
                e.payload.slotPair.uInfo =
                    EmitAD(EBcOp::UGet, 0, static_cast<std::uint16_t>(e.payload.slotPair.uInfo));
                e.eKind = EExpKind::Relocatable;
                break;
            case EExpKind::Global:
                e.payload.slotPair.uInfo = EmitAD(
                    EBcOp::GGet, 0,
                    static_cast<std::uint16_t>(StrConst(
                        static_cast<C_GcString*>(e.payload.tvValue.AsGcPointer()))));
                e.eKind = EExpKind::Relocatable;
                break;
            case EExpKind::Indexed: {
                const std::uint32_t uTableReg = e.payload.slotPair.uInfo;
                const std::uint32_t uKey = e.payload.slotPair.uAux;
                std::uint32_t uPos;
                if (uKey >= kKeyByteBase) {
                    uPos = EmitABC(EBcOp::TGetB, 0, static_cast<std::uint8_t>(uTableReg),
                                   static_cast<std::uint8_t>(uKey - kKeyByteBase));
                } else if (uKey >= kKeyStrBase) {
                    uPos = EmitABC(EBcOp::TGetS, 0, static_cast<std::uint8_t>(uTableReg),
                                   static_cast<std::uint8_t>(uKey - kKeyStrBase));
                } else {
                    FreeReg(uKey);
                    uPos = EmitABC(EBcOp::TGetV, 0, static_cast<std::uint8_t>(uTableReg),
                                   static_cast<std::uint8_t>(uKey));
                }
                FreeReg(uTableReg);
                e.payload.slotPair.uInfo = uPos;
                e.eKind = EExpKind::Relocatable;
                break;
            }
            case EExpKind::Call: {
                // One result: it lands at the call base.
                e.payload.slotPair.uInfo = e.payload.slotPair.uAux;
                e.eKind = EExpKind::NonReloc;
                break;
            }
            default: break;
        }
    }

    void Discharge2Reg(ExpDesc_t& e, std::uint8_t uReg) {
        DischargeVars(e);
        switch (e.eKind) {
            case EExpKind::KNil: EmitAD(EBcOp::KNil, uReg, uReg); break;
            case EExpKind::KFalse: EmitAD(EBcOp::KPri, uReg, 1); break;
            case EExpKind::KTrue: EmitAD(EBcOp::KPri, uReg, 2); break;
            case EExpKind::KString:
                EmitAD(EBcOp::KStr, uReg,
                       static_cast<std::uint16_t>(StrConst(
                           static_cast<C_GcString*>(e.payload.tvValue.AsGcPointer()))));
                break;
            case EExpKind::KNumber: {
                const double flValue = e.payload.tvValue.AsDouble();
                std::int32_t nValue;
                if (core::NumToInt32Check(flValue, nValue) && nValue >= -0x8000 &&
                    nValue <= 0x7fff)
                    EmitAD(EBcOp::KShort, uReg,
                           static_cast<std::uint16_t>(static_cast<std::int16_t>(nValue)));
                else
                    EmitAD(EBcOp::KNum, uReg, static_cast<std::uint16_t>(NumConst(flValue)));
                break;
            }
            case EExpKind::Relocatable: {
                BcIns_t& ins = fs->vCode[e.payload.slotPair.uInfo];
                ins.uRaw = (ins.uRaw & ~0xff00u) | (std::uint32_t{uReg} << 8);
                break;
            }
            case EExpKind::NonReloc:
                if (e.payload.slotPair.uInfo != uReg)
                    EmitAD(EBcOp::Mov, uReg,
                           static_cast<std::uint16_t>(e.payload.slotPair.uInfo));
                break;
            case EExpKind::Jump: break;  // handled by Exp2Reg
            default: Error("internal: cannot discharge expression");
        }
        if (e.eKind != EExpKind::Jump) {
            e.payload.slotPair.uInfo = uReg;
            e.eKind = EExpKind::NonReloc;
        }
    }

    void Exp2Reg(ExpDesc_t& e, std::uint8_t uReg) {
        Discharge2Reg(e, uReg);
        if (e.eKind == EExpKind::Jump) Concat(e.posTrueList, e.payload.slotPair.uInfo);
        if (e.HasJumps()) {
            std::uint32_t posFalseLabel = kNoJump;
            std::uint32_t posTrueLabel = kNoJump;
            if (ListNeedsValue(e.posTrueList) || ListNeedsValue(e.posFalseList) ||
                e.eKind == EExpKind::Jump) {
                const std::uint32_t posSkip =
                    e.eKind == EExpKind::Jump ? kNoJump : EmitJump();
                posFalseLabel = EmitAD(EBcOp::KPri, uReg, 1);   // false
                const std::uint32_t posOver = EmitJump();
                posTrueLabel = EmitAD(EBcOp::KPri, uReg, 2);    // true
                if (posSkip != kNoJump) {
                    FixJump(posSkip, Pos());
                }
                FixJump(posOver, Pos());
            }
            const std::uint32_t uFinal = Pos();
            PatchListAux(e.posFalseList, uFinal, uReg, posFalseLabel);
            PatchListAux(e.posTrueList, uFinal, uReg, posTrueLabel);
        }
        e.posTrueList = e.posFalseList = kNoJump;
        e.payload.slotPair.uInfo = uReg;
        e.eKind = EExpKind::NonReloc;
    }

    void Exp2NextReg(ExpDesc_t& e) {
        DischargeVars(e);
        FreeExp(e);
        ReserveRegs(1);
        Exp2Reg(e, static_cast<std::uint8_t>(fs->uFreeReg - 1));
    }

    std::uint8_t Exp2AnyReg(ExpDesc_t& e) {
        DischargeVars(e);
        if (e.eKind == EExpKind::NonReloc && !e.HasJumps())
            return static_cast<std::uint8_t>(e.payload.slotPair.uInfo);
        Exp2NextReg(e);
        return static_cast<std::uint8_t>(e.payload.slotPair.uInfo);
    }

    void Exp2Val(ExpDesc_t& e) {
        if (e.HasJumps() || e.eKind == EExpKind::Jump)
            Exp2AnyReg(e);
        else
            DischargeVars(e);
    }

    // ---- conditions --------------------------------------------------------
    void InvertCond(ExpDesc_t& e) {
        BcIns_t& ins = fs->vCode[JumpControlPos(e.payload.slotPair.uInfo)];
        ins.uRaw ^= 1;  // opcode algebra: comparison XOR 1 flips the condition
    }
    std::uint32_t JumpOnCond(ExpDesc_t& e, bool bJumpWhenTrue) {
        Discharge2AnyRegKeep(e);
        FreeExp(e);
        EmitAD(bJumpWhenTrue ? EBcOp::IsTC : EBcOp::IsFC, kNoRegSentinel,
               static_cast<std::uint16_t>(e.payload.slotPair.uInfo));
        return EmitJump();
    }
    void Discharge2AnyRegKeep(ExpDesc_t& e) {
        DischargeVars(e);
        if (e.eKind != EExpKind::NonReloc) {
            ReserveRegs(1);
            Discharge2Reg(e, static_cast<std::uint8_t>(fs->uFreeReg - 1));
        }
    }
    void GoIfTrue(ExpDesc_t& e) {  // jump when FALSE, fall through when true
        DischargeVars(e);
        std::uint32_t uPos;
        switch (e.eKind) {
            case EExpKind::KTrue: case EExpKind::KNumber: case EExpKind::KString:
                uPos = kNoJump;
                break;
            case EExpKind::Jump:
                InvertCond(e);
                uPos = e.payload.slotPair.uInfo;
                break;
            default:
                uPos = JumpOnCond(e, false);
                break;
        }
        Concat(e.posFalseList, uPos);
        PatchToHere(e.posTrueList);
        e.posTrueList = kNoJump;
    }
    void GoIfFalse(ExpDesc_t& e) {  // jump when TRUE
        DischargeVars(e);
        std::uint32_t uPos;
        switch (e.eKind) {
            case EExpKind::KNil: case EExpKind::KFalse:
                uPos = kNoJump;
                break;
            case EExpKind::Jump:
                uPos = e.payload.slotPair.uInfo;
                break;
            default:
                uPos = JumpOnCond(e, true);
                break;
        }
        Concat(e.posTrueList, uPos);
        PatchToHere(e.posFalseList);
        e.posFalseList = kNoJump;
    }
    void CodeNot(ExpDesc_t& e) {
        DischargeVars(e);
        switch (e.eKind) {
            case EExpKind::KNil: case EExpKind::KFalse: e.eKind = EExpKind::KTrue; break;
            case EExpKind::KTrue: case EExpKind::KNumber: case EExpKind::KString:
                e.eKind = EExpKind::KFalse;
                break;
            case EExpKind::Jump: InvertCond(e); break;
            default: {
                Discharge2AnyRegKeep(e);
                FreeExp(e);
                e.payload.slotPair.uInfo =
                    EmitAD(EBcOp::Not, 0,
                           static_cast<std::uint16_t>(e.payload.slotPair.uInfo));
                e.eKind = EExpKind::Relocatable;
                break;
            }
        }
        std::swap(e.posTrueList, e.posFalseList);
        // Values in the swapped lists no longer matter (result is a boolean).
    }

    // ---- variables ---------------------------------------------------------
    [[nodiscard]] std::int32_t SearchLocal(FuncState_t* pFunc, C_GcString* pName) const {
        for (std::int32_t nI = static_cast<std::int32_t>(pFunc->uNumActive) - 1; nI >= 0; --nI)
            if (pFunc->vLocals[static_cast<std::uint32_t>(nI)] == pName) return nI;
        return -1;
    }
    std::uint32_t AddUpvalue(FuncState_t* pFunc, C_GcString* pName, bool bFromLocal,
                             std::uint8_t uIndex) {
        for (std::uint32_t uI = 0; uI < pFunc->vUpvals.size(); ++uI) {
            const UpvalDesc_t& desc = pFunc->vUpvals[uI];
            if (desc.pName == pName && desc.bFromParentLocal == bFromLocal &&
                desc.uIndex == uIndex)
                return uI;
        }
        if (pFunc->vUpvals.size() >= 60) Error("too many upvalues");
        pFunc->vUpvals.push_back({pName, bFromLocal, uIndex});
        return static_cast<std::uint32_t>(pFunc->vUpvals.size() - 1);
    }
    // Resolves in pFunc; kind Local/Upvalue/Global.
    EExpKind ResolveVar(FuncState_t* pFunc, C_GcString* pName, ExpDesc_t& e) {
        const std::int32_t nSlot = SearchLocal(pFunc, pName);
        if (nSlot >= 0) {
            e.eKind = EExpKind::Local;
            e.payload.slotPair.uInfo = static_cast<std::uint32_t>(nSlot);
            return EExpKind::Local;
        }
        if (!pFunc->pParent) return EExpKind::Global;
        const EExpKind eParentKind = ResolveVar(pFunc->pParent, pName, e);
        if (eParentKind == EExpKind::Global) return EExpKind::Global;
        const std::uint8_t uParentIndex = static_cast<std::uint8_t>(e.payload.slotPair.uInfo);
        if (eParentKind == EExpKind::Local) pFunc->pParent->bAnyCaptured = true;
        e.payload.slotPair.uInfo =
            AddUpvalue(pFunc, pName, eParentKind == EExpKind::Local, uParentIndex);
        e.eKind = EExpKind::Upvalue;
        return EExpKind::Upvalue;
    }
    void SingleVar(ExpDesc_t& e, C_GcString* pName) {
        e = ExpDesc_t{};
        if (ResolveVar(fs, pName, e) == EExpKind::Global) {
            e.eKind = EExpKind::Global;
            e.payload.tvValue = TValue_t::GcObject(vm::EValueTag::String, pName);
        }
    }
    void NewLocal(C_GcString* pName) {
        // Invariant maintained by ActivateLocals / LeaveBlock: at the start of
        // any local-declaration batch vLocals.size() == uNumActive, so a plain
        // append places names contiguously above the active window.
        fs->vLocals.push_back(pName);
    }
    void ActivateLocals(std::uint32_t uCount) {
        fs->uNumActive += uCount;
        if (fs->uNumActive > fs->uFreeReg) fs->uFreeReg = fs->uNumActive;
        if (fs->uFreeReg + 2 > fs->uFrameMax)
            fs->uFrameMax = fs->uFreeReg + 2;
    }

    // ---- assignments -------------------------------------------------------
    void StoreVar(const ExpDesc_t& target, ExpDesc_t& value) {
        switch (target.eKind) {
            case EExpKind::Local: {
                FreeExp(value);
                Exp2Reg(value, static_cast<std::uint8_t>(target.payload.slotPair.uInfo));
                return;
            }
            case EExpKind::Upvalue: {
                const std::uint8_t uReg = Exp2AnyReg(value);
                EmitAD(EBcOp::USetV, static_cast<std::uint8_t>(target.payload.slotPair.uInfo),
                       uReg);
                break;
            }
            case EExpKind::Global: {
                const std::uint8_t uReg = Exp2AnyReg(value);
                EmitAD(EBcOp::GSet, uReg,
                       static_cast<std::uint16_t>(StrConst(
                           static_cast<C_GcString*>(target.payload.tvValue.AsGcPointer()))));
                break;
            }
            case EExpKind::Indexed: {
                const std::uint8_t uReg = Exp2AnyReg(value);
                const std::uint32_t uKey = target.payload.slotPair.uAux;
                if (uKey >= kKeyByteBase)
                    EmitABC(EBcOp::TSetB, uReg,
                            static_cast<std::uint8_t>(target.payload.slotPair.uInfo),
                            static_cast<std::uint8_t>(uKey - kKeyByteBase));
                else if (uKey >= kKeyStrBase)
                    EmitABC(EBcOp::TSetS, uReg,
                            static_cast<std::uint8_t>(target.payload.slotPair.uInfo),
                            static_cast<std::uint8_t>(uKey - kKeyStrBase));
                else
                    EmitABC(EBcOp::TSetV, uReg,
                            static_cast<std::uint8_t>(target.payload.slotPair.uInfo),
                            static_cast<std::uint8_t>(uKey));
                break;
            }
            default: Error("cannot assign to this expression");
        }
        FreeExp(value);
    }

    // ---- indexed access ----------------------------------------------------
    void IndexedKey(ExpDesc_t& eTable, ExpDesc_t& eKey) {
        const std::uint8_t uTableReg = Exp2AnyReg(eTable);
        eTable.eKind = EExpKind::Indexed;
        eTable.payload.slotPair.uInfo = uTableReg;
        if (eKey.eKind == EExpKind::KString) {
            const std::uint32_t uIdx =
                StrConst(static_cast<C_GcString*>(eKey.payload.tvValue.AsGcPointer()));
            if (uIdx <= 0xff) {
                eTable.payload.slotPair.uAux = kKeyStrBase + uIdx;
                return;
            }
        } else if (eKey.eKind == EExpKind::KNumber) {
            const double flKey = eKey.payload.tvValue.AsDouble();
            std::int32_t nKey;
            if (core::NumToInt32Check(flKey, nKey) && nKey >= 0 && nKey <= 0xff) {
                eTable.payload.slotPair.uAux = kKeyByteBase + static_cast<std::uint32_t>(nKey);
                return;
            }
        }
        eTable.payload.slotPair.uAux = Exp2AnyReg(eKey);
    }

    // (Parsing methods follow in part 2 of this file.)
    void ParseChunkBody();
    C_GcProto* FinishFunction();
    void ParseBlock();
    bool ParseStatement();
    void ParseExpr(ExpDesc_t& e);
    void ParseSubExpr(ExpDesc_t& e, std::uint32_t uLimit);
    void ParseSimpleExpr(ExpDesc_t& e);
    void ParseSuffixedExpr(ExpDesc_t& e);
    void ParsePrimaryExpr(ExpDesc_t& e);
    void ParseCallArgs(ExpDesc_t& eFunc);
    void ParseTableConstructor(ExpDesc_t& e);
    void ParseFunctionBody(ExpDesc_t& e, bool bIsMethod);
    std::uint32_t ParseExprList(ExpDesc_t& eLast);
    void AdjustAssign(std::uint32_t uWanted, std::uint32_t uGiven, ExpDesc_t& eLast);
    void ParseAssignmentOrCall();
    void ParseLocal();
    void ParseIf();
    void ParseWhile();
    void ParseRepeat();
    void ParseFor();
    void ParseFunctionStatement();
    void ParseReturn();
    void EnterBlock(BlockScope_t& block, bool bIsLoop);
    void LeaveBlock();
    void EmitCloseIfCaptured(std::uint32_t uFromSlot);
};

}  // namespace

// ---------------------------------------------------------------------------
// Parsing (part 2) — statements, expressions, function assembly.
// ---------------------------------------------------------------------------

namespace {

// Binding powers (left/right) per binary operator token.
struct BinPower_t {
    std::uint8_t uLeft, uRight;
};
constexpr std::uint32_t kUnaryPower = 8;

bool BinaryPower(ETokenKind eTok, BinPower_t& power, EBcOp& eOpVV, int& nKind) {
    switch (static_cast<std::uint16_t>(eTok)) {
        case '+': power = {6, 6}; eOpVV = EBcOp::AddVV; nKind = 0; return true;
        case '-': power = {6, 6}; eOpVV = EBcOp::SubVV; nKind = 0; return true;
        case '*': power = {7, 7}; eOpVV = EBcOp::MulVV; nKind = 0; return true;
        case '/': power = {7, 7}; eOpVV = EBcOp::DivVV; nKind = 0; return true;
        case '%': power = {7, 7}; eOpVV = EBcOp::ModVV; nKind = 0; return true;
        case '^': power = {10, 9}; eOpVV = EBcOp::Pow; nKind = 1; return true;   // right assoc
        case static_cast<std::uint16_t>(ETokenKind::Concat): power = {5, 4}; eOpVV = EBcOp::Cat; nKind = 2; return true;             // right assoc
        case static_cast<std::uint16_t>(ETokenKind::Eq): power = {3, 3}; eOpVV = EBcOp::IsEqV; nKind = 3; return true;
        case static_cast<std::uint16_t>(ETokenKind::Ne): power = {3, 3}; eOpVV = EBcOp::IsNeV; nKind = 3; return true;
        case '<': power = {3, 3}; eOpVV = EBcOp::IsLt; nKind = 4; return true;
        case static_cast<std::uint16_t>(ETokenKind::Le): power = {3, 3}; eOpVV = EBcOp::IsLe; nKind = 4; return true;
        case '>': power = {3, 3}; eOpVV = EBcOp::IsGt; nKind = 4; return true;
        case static_cast<std::uint16_t>(ETokenKind::Ge): power = {3, 3}; eOpVV = EBcOp::IsGe; nKind = 4; return true;
        case static_cast<std::uint16_t>(ETokenKind::And): power = {2, 2}; eOpVV = EBcOp::Mov; nKind = 5; return true;
        case static_cast<std::uint16_t>(ETokenKind::Or): power = {1, 1}; eOpVV = EBcOp::Mov; nKind = 6; return true;
        default: return false;
    }
}

double FoldArith(EBcOp eOp, double flA, double flB) {
    switch (eOp) {
        case EBcOp::AddVV: return flA + flB;
        case EBcOp::SubVV: return flA - flB;
        case EBcOp::MulVV: return flA * flB;
        case EBcOp::DivVV: return flA / flB;
        case EBcOp::ModVV: {
            const double flMod = flA - __builtin_floor(flA / flB) * flB;
            return flMod;
        }
        case EBcOp::Pow: return __builtin_pow(flA, flB);
        default: return 0.0;
    }
}

}  // namespace

// NOTE: member definitions must live inside the anonymous namespace where
// Ctx_t is defined — reopen it.
namespace {

void Ctx_t::EnterBlock(BlockScope_t& block, bool bIsLoop) {
    block.pPrev = fs->pBlock;
    block.uFirstLocal = fs->uNumActive;
    block.posBreakList = kNoJump;
    block.bIsLoop = bIsLoop;
    fs->pBlock = &block;
}

void Ctx_t::EmitCloseIfCaptured(std::uint32_t uFromSlot) {
    if (fs->bAnyCaptured)
        EmitAD(EBcOp::UClo, static_cast<std::uint8_t>(uFromSlot), vm::kJumpBias);
}

void Ctx_t::LeaveBlock() {
    BlockScope_t* pBlock = fs->pBlock;
    fs->pBlock = pBlock->pPrev;
    if (fs->uNumActive > pBlock->uFirstLocal) EmitCloseIfCaptured(pBlock->uFirstLocal);
    fs->uNumActive = pBlock->uFirstLocal;
    fs->vLocals.resize(fs->uNumActive);
    fs->uFreeReg = fs->uNumActive;
    if (pBlock->posBreakList != kNoJump) PatchToHere(pBlock->posBreakList);
}

void Ctx_t::ParseExpr(ExpDesc_t& e) { ParseSubExpr(e, 0); }

void Ctx_t::ParseSimpleExpr(ExpDesc_t& e) {
    e = ExpDesc_t{};
    switch (static_cast<std::uint16_t>(Tok())) {
        case static_cast<std::uint16_t>(ETokenKind::Number):
            e.eKind = EExpKind::KNumber;
            e.payload.tvValue = lex.Current().tvValue;
            Next();
            return;
        case static_cast<std::uint16_t>(ETokenKind::String):
            e.eKind = EExpKind::KString;
            e.payload.tvValue = lex.Current().tvValue;
            Next();
            return;
        case static_cast<std::uint16_t>(ETokenKind::Nil): e.eKind = EExpKind::KNil; Next(); return;
        case static_cast<std::uint16_t>(ETokenKind::True): e.eKind = EExpKind::KTrue; Next(); return;
        case static_cast<std::uint16_t>(ETokenKind::False): e.eKind = EExpKind::KFalse; Next(); return;
        case '{': ParseTableConstructor(e); return;
        case static_cast<std::uint16_t>(ETokenKind::Function):
            Next();
            ParseFunctionBody(e, false);
            return;
        case static_cast<std::uint16_t>(ETokenKind::Ellipsis): Error("'...' is not supported yet");
        default: ParseSuffixedExpr(e); return;
    }
}

void Ctx_t::ParseSubExpr(ExpDesc_t& e, std::uint32_t uLimit) {
    // Unary operators.
    if (Tok() == ETokenKind::Not) {
        Next();
        ParseSubExpr(e, kUnaryPower);
        CodeNot(e);
    } else if (Tok() == static_cast<ETokenKind>('-')) {
        Next();
        ParseSubExpr(e, kUnaryPower);
        if (e.eKind == EExpKind::KNumber) {
            e.payload.tvValue = TValue_t::Number(-e.payload.tvValue.AsDouble());
        } else {
            Discharge2AnyRegKeep(e);
            FreeExp(e);
            e.payload.slotPair.uInfo =
                EmitAD(EBcOp::Unm, 0, static_cast<std::uint16_t>(e.payload.slotPair.uInfo));
            e.eKind = EExpKind::Relocatable;
        }
    } else if (Tok() == static_cast<ETokenKind>('#')) {
        Next();
        ParseSubExpr(e, kUnaryPower);
        Discharge2AnyRegKeep(e);
        FreeExp(e);
        e.payload.slotPair.uInfo =
            EmitAD(EBcOp::Len, 0, static_cast<std::uint16_t>(e.payload.slotPair.uInfo));
        e.eKind = EExpKind::Relocatable;
    } else {
        ParseSimpleExpr(e);
    }

    // Binary operators by precedence climbing.
    for (;;) {
        BinPower_t power;
        EBcOp eOpVV;
        int nKind;
        if (!BinaryPower(Tok(), power, eOpVV, nKind) || power.uLeft <= uLimit) return;
        Next();

        if (nKind == 5) {  // and
            GoIfTrue(e);
            ExpDesc_t e2;
            const std::uint32_t posFalse = e.posFalseList;
            ParseSubExpr(e2, power.uRight);
            Concat(e2.posFalseList, posFalse);
            e = e2;
            continue;
        }
        if (nKind == 6) {  // or
            GoIfFalse(e);
            ExpDesc_t e2;
            const std::uint32_t posTrue = e.posTrueList;
            ParseSubExpr(e2, power.uRight);
            Concat(e2.posTrueList, posTrue);
            e = e2;
            continue;
        }
        if (nKind == 2) {  // concat (right assoc, chain fusion)
            Exp2NextReg(e);
            ExpDesc_t e2;
            ParseSubExpr(e2, power.uRight);
            if (e2.eKind == EExpKind::Relocatable &&
                fs->vCode[e2.payload.slotPair.uInfo].Op() == EBcOp::Cat) {
                BcIns_t& ins = fs->vCode[e2.payload.slotPair.uInfo];
                // Extend the run downward: B := our slot, then release it —
                // the next reservation becomes the Cat's own destination.
                const std::uint8_t uOurReg =
                    static_cast<std::uint8_t>(e.payload.slotPair.uInfo);
                ins.uRaw = (ins.uRaw & ~0xff000000u) | (std::uint32_t{uOurReg} << 24);
                FreeExp(e);
                e.payload.slotPair.uInfo = e2.payload.slotPair.uInfo;
                e.eKind = EExpKind::Relocatable;
            } else {
                Exp2NextReg(e2);
                const std::uint8_t uFirst = static_cast<std::uint8_t>(e.payload.slotPair.uInfo);
                const std::uint8_t uLast = static_cast<std::uint8_t>(e2.payload.slotPair.uInfo);
                FreeExp(e2);
                FreeExp(e);
                e.payload.slotPair.uInfo = EmitABC(EBcOp::Cat, 0, uFirst, uLast);
                e.eKind = EExpKind::Relocatable;
            }
            continue;
        }
        if (nKind == 3 || nKind == 4) {  // comparisons
            Exp2Val(e);
            // Pin the left operand into a register before parsing the right
            // one — see the note in the arithmetic case below.
            if (!e.IsConstant()) Exp2AnyReg(e);
            ExpDesc_t e2;
            ParseSubExpr(e2, power.uRight);
            Exp2Val(e2);
            std::uint32_t uPos;
            if (nKind == 3 && e2.eKind == EExpKind::KString) {
                const std::uint8_t uReg = Exp2AnyReg(e);
                FreeExp(e);
                uPos = EmitAD(eOpVV == EBcOp::IsEqV ? EBcOp::IsEqS : EBcOp::IsNeS, uReg,
                              static_cast<std::uint16_t>(StrConst(static_cast<C_GcString*>(
                                  e2.payload.tvValue.AsGcPointer()))));
            } else if (nKind == 3 && e2.eKind == EExpKind::KNumber) {
                const std::uint8_t uReg = Exp2AnyReg(e);
                FreeExp(e);
                uPos = EmitAD(eOpVV == EBcOp::IsEqV ? EBcOp::IsEqN : EBcOp::IsNeN, uReg,
                              static_cast<std::uint16_t>(
                                  NumConst(e2.payload.tvValue.AsDouble())));
            } else if (nKind == 3 && e2.eKind <= EExpKind::KTrue) {
                const std::uint8_t uReg = Exp2AnyReg(e);
                FreeExp(e);
                uPos = EmitAD(eOpVV == EBcOp::IsEqV ? EBcOp::IsEqP : EBcOp::IsNeP, uReg,
                              static_cast<std::uint16_t>(e2.eKind));
            } else {
                const std::uint8_t uRegB = Exp2AnyReg(e);
                const std::uint8_t uRegD = Exp2AnyReg(e2);
                FreeExp(e2);
                FreeExp(e);
                uPos = EmitAD(eOpVV, uRegB, uRegD);
            }
            e = ExpDesc_t{};
            e.eKind = EExpKind::Jump;
            e.payload.slotPair.uInfo = EmitJump();
            (void)uPos;
            continue;
        }
        // Arithmetic / pow.
        if (nKind == 0 && e.IsConstant() && e.eKind == EExpKind::KNumber) {
            // Maybe fold with the RHS below.
        }
        ExpDesc_t e2;
        if (nKind == 0) {
            Exp2Val(e);
            // Materialize the left operand into a register BEFORE parsing the
            // right one. A discharged-but-relocatable expression (UGet, GGet,
            // TGet*, an arithmetic result) has emitted its instruction but
            // holds NO register reservation, so anything the right-hand side
            // allocates lands on the same slot and clobbers it — e.g.
            // `c + w[i]` would emit UGet r4,c / UGet r4,w. Numeric constants
            // are deliberately left unmaterialized so the NV operand-kind
            // variant can still fold them into the instruction.
            if (e.eKind != EExpKind::KNumber) Exp2AnyReg(e);
            ParseSubExpr(e2, power.uRight);
            Exp2Val(e2);
            // Parse-time folding (never to NaN or -0: constant-table hygiene).
            if (e.eKind == EExpKind::KNumber && e2.eKind == EExpKind::KNumber) {
                const double flResult =
                    FoldArith(eOpVV, e.payload.tvValue.AsDouble(),
                              e2.payload.tvValue.AsDouble());
                const bool bIsNan = flResult != flResult;
                const bool bIsNegZero =
                    flResult == 0.0 && std::bit_cast<std::uint64_t>(flResult) != 0;
                if (!bIsNan && !bIsNegZero) {
                    e.payload.tvValue = TValue_t::Number(flResult);
                    continue;
                }
            }
            // Operand-kind variants: VN (const rhs), NV (const lhs), else VV.
            const auto uOpBase = static_cast<std::uint8_t>(eOpVV);
            if (e2.eKind == EExpKind::KNumber) {
                const std::uint32_t uIdx = NumConst(e2.payload.tvValue.AsDouble());
                if (uIdx <= 0xff) {
                    const std::uint8_t uRegB = Exp2AnyReg(e);
                    FreeExp(e);
                    const auto eOpVN = static_cast<EBcOp>(
                        uOpBase - (static_cast<std::uint8_t>(EBcOp::AddVV) -
                                   static_cast<std::uint8_t>(EBcOp::AddVN)));
                    e.payload.slotPair.uInfo =
                        EmitABC(eOpVN, 0, uRegB, static_cast<std::uint8_t>(uIdx));
                    e.eKind = EExpKind::Relocatable;
                    continue;
                }
            }
            if (e.eKind == EExpKind::KNumber) {
                const std::uint32_t uIdx = NumConst(e.payload.tvValue.AsDouble());
                if (uIdx <= 0xff) {
                    const std::uint8_t uRegB = Exp2AnyReg(e2);
                    FreeExp(e2);
                    const auto eOpNV = static_cast<EBcOp>(
                        uOpBase - (static_cast<std::uint8_t>(EBcOp::AddVV) -
                                   static_cast<std::uint8_t>(EBcOp::AddNV)));
                    e.payload.slotPair.uInfo =
                        EmitABC(eOpNV, 0, uRegB, static_cast<std::uint8_t>(uIdx));
                    e.eKind = EExpKind::Relocatable;
                    continue;
                }
            }
        } else {  // pow: both to registers
            Exp2Val(e);
            if (!e.IsConstant()) Exp2AnyReg(e);
            ParseSubExpr(e2, power.uRight);
        }
        const std::uint8_t uRegB = Exp2AnyReg(e);
        const std::uint8_t uRegC = Exp2AnyReg(e2);
        FreeExp(e2);
        FreeExp(e);
        e.payload.slotPair.uInfo = EmitABC(eOpVV, 0, uRegB, uRegC);
        e.eKind = EExpKind::Relocatable;
        e.posTrueList = e.posFalseList = kNoJump;
    }
}

void Ctx_t::ParsePrimaryExpr(ExpDesc_t& e) {
    if (Tok() == ETokenKind::Name) {
        SingleVar(e, ExpectName());
        return;
    }
    if (Tok() == static_cast<ETokenKind>('(')) {
        Next();
        ParseExpr(e);
        Expect(static_cast<ETokenKind>(')'), ")");
        // Parenthesized expressions truncate to one value.
        Exp2Val(e);
        if (e.eKind == EExpKind::Call) DischargeVars(e);
        return;
    }
    Error("unexpected symbol");
}

void Ctx_t::ParseCallArgs(ExpDesc_t& eFunc) {
    // eFunc value must already sit at a fresh register (the call base);
    // reserve the frame-link slot, then the arguments.
    const std::uint8_t uBase = static_cast<std::uint8_t>(eFunc.payload.slotPair.uInfo);
    ReserveRegs(1);  // frame-link slot at base+1
    std::uint32_t uArgCount = 0;
    bool bMultiRet = false;
    if (Tok() == static_cast<ETokenKind>('(')) {
        Next();
        if (Tok() != static_cast<ETokenKind>(')')) {
            for (;;) {
                ExpDesc_t eArg;
                ParseExpr(eArg);
                ++uArgCount;
                if (Tok() == static_cast<ETokenKind>(',')) {
                    Exp2NextReg(eArg);
                    Next();
                    continue;
                }
                // Last argument: calls expand to all results.
                if (eArg.eKind == EExpKind::Call) {
                    BcIns_t& ins = fs->vCode[eArg.payload.slotPair.uInfo];
                    ins.uRaw = (ins.uRaw & ~0xff000000u);  // B = 0: all results
                    bMultiRet = true;
                    --uArgCount;
                } else {
                    Exp2NextReg(eArg);
                }
                break;
            }
        }
        Expect(static_cast<ETokenKind>(')'), ")");
    } else if (Tok() == ETokenKind::String) {
        ExpDesc_t eArg;
        eArg.eKind = EExpKind::KString;
        eArg.payload.tvValue = lex.Current().tvValue;
        Next();
        Exp2NextReg(eArg);
        uArgCount = 1;
    } else if (Tok() == static_cast<ETokenKind>('{')) {
        ExpDesc_t eArg;
        ParseTableConstructor(eArg);
        Exp2NextReg(eArg);
        uArgCount = 1;
    } else {
        Error("function arguments expected");
    }
    const std::uint32_t uPos =
        bMultiRet ? EmitABC(EBcOp::CallM, uBase, 2, static_cast<std::uint8_t>(uArgCount))
                  : EmitABC(EBcOp::Call, uBase, 2, static_cast<std::uint8_t>(uArgCount + 1));
    eFunc = ExpDesc_t{};
    eFunc.eKind = EExpKind::Call;
    eFunc.payload.slotPair.uInfo = uPos;
    eFunc.payload.slotPair.uAux = uBase;
    fs->uFreeReg = uBase + 1;  // one result by default
}

void Ctx_t::ParseSuffixedExpr(ExpDesc_t& e) {
    ParsePrimaryExpr(e);
    for (;;) {
        switch (static_cast<std::uint16_t>(Tok())) {
            case '.': {
                Next();
                C_GcString* pField = ExpectName();
                ExpDesc_t eKey;
                eKey.eKind = EExpKind::KString;
                eKey.payload.tvValue = TValue_t::GcObject(vm::EValueTag::String, pField);
                IndexedKey(e, eKey);
                break;
            }
            case '[': {
                Next();
                ExpDesc_t eKey;
                ParseExpr(eKey);
                Exp2Val(eKey);
                Expect(static_cast<ETokenKind>(']'), "]");
                IndexedKey(e, eKey);
                break;
            }
            case ':': {
                Next();
                C_GcString* pMethod = ExpectName();
                // Method call: t at some reg; layout func/link/self then args.
                const std::uint8_t uObjReg = Exp2AnyReg(e);
                FreeExp(e);
                const std::uint8_t uBase = static_cast<std::uint8_t>(fs->uFreeReg);
                ReserveRegs(3);  // func, link, self
                EmitAD(EBcOp::Mov, static_cast<std::uint8_t>(uBase + 2), uObjReg);
                const std::uint32_t uIdx = StrConst(pMethod);
                if (uIdx <= 0xff) {
                    EmitABC(EBcOp::TGetS, uBase, static_cast<std::uint8_t>(uBase + 2),
                            static_cast<std::uint8_t>(uIdx));
                } else {
                    Error("too many method-name constants");
                }
                // Now parse args with self prepended: reuse ParseCallArgs's
                // shape but the base/link/self are already reserved.
                std::uint32_t uArgCount = 1;  // self
                bool bMultiRet = false;
                if (Tok() == static_cast<ETokenKind>('(')) {
                    Next();
                    if (Tok() != static_cast<ETokenKind>(')')) {
                        for (;;) {
                            ExpDesc_t eArg;
                            ParseExpr(eArg);
                            ++uArgCount;
                            if (Tok() == static_cast<ETokenKind>(',')) {
                                Exp2NextReg(eArg);
                                Next();
                                continue;
                            }
                            if (eArg.eKind == EExpKind::Call) {
                                fs->vCode[eArg.payload.slotPair.uInfo].uRaw &= ~0xff000000u;
                                bMultiRet = true;
                                --uArgCount;
                            } else {
                                Exp2NextReg(eArg);
                            }
                            break;
                        }
                    }
                    Expect(static_cast<ETokenKind>(')'), ")");
                } else if (Tok() == ETokenKind::String) {
                    ExpDesc_t eArg;
                    eArg.eKind = EExpKind::KString;
                    eArg.payload.tvValue = lex.Current().tvValue;
                    Next();
                    Exp2NextReg(eArg);
                    uArgCount = 2;
                } else {
                    Error("function arguments expected");
                }
                const std::uint32_t uPos =
                    bMultiRet
                        ? EmitABC(EBcOp::CallM, uBase, 2, static_cast<std::uint8_t>(uArgCount))
                        : EmitABC(EBcOp::Call, uBase, 2,
                                  static_cast<std::uint8_t>(uArgCount + 1));
                e = ExpDesc_t{};
                e.eKind = EExpKind::Call;
                e.payload.slotPair.uInfo = uPos;
                e.payload.slotPair.uAux = uBase;
                fs->uFreeReg = uBase + 1;
                break;
            }
            case '(': case '{': {
                Exp2NextReg(e);
                ParseCallArgs(e);
                break;
            }
            default:
                if (Tok() == ETokenKind::String) {
                    Exp2NextReg(e);
                    ParseCallArgs(e);
                    break;
                }
                return;
        }
    }
}

void Ctx_t::ParseTableConstructor(ExpDesc_t& e) {
    Expect(static_cast<ETokenKind>('{'), "{");
    const std::uint8_t uTableReg = static_cast<std::uint8_t>(fs->uFreeReg);
    const std::uint32_t uTNewPos = EmitAD(EBcOp::TNew, uTableReg, 0);
    ReserveRegs(1);
    e = ExpDesc_t{};
    e.eKind = EExpKind::NonReloc;
    e.payload.slotPair.uInfo = uTableReg;

    std::uint32_t uArrayIndex = 1;
    std::uint32_t uHashCount = 0;
    while (Tok() != static_cast<ETokenKind>('}')) {
        if (Tok() == ETokenKind::Name &&
            lex.Lookahead().eKind == static_cast<ETokenKind>('=')) {
            C_GcString* pKey = ExpectName();
            Next();  // '='
            ExpDesc_t eValue;
            ParseExpr(eValue);
            const std::uint8_t uValReg = Exp2AnyReg(eValue);
            const std::uint32_t uIdx = StrConst(pKey);
            if (uIdx > 0xff) Error("too many table-key constants");
            EmitABC(EBcOp::TSetS, uValReg, uTableReg, static_cast<std::uint8_t>(uIdx));
            FreeExp(eValue);
            ++uHashCount;
        } else if (Tok() == static_cast<ETokenKind>('[')) {
            Next();
            ExpDesc_t eKey;
            ParseExpr(eKey);
            Expect(static_cast<ETokenKind>(']'), "]");
            Expect(static_cast<ETokenKind>('='), "=");
            const std::uint8_t uKeyReg = Exp2AnyReg(eKey);
            ExpDesc_t eValue;
            ParseExpr(eValue);
            const std::uint8_t uValReg = Exp2AnyReg(eValue);
            EmitABC(EBcOp::TSetV, uValReg, uTableReg, uKeyReg);
            FreeExp(eValue);
            FreeExp(eKey);
            ++uHashCount;
        } else {
            ExpDesc_t eValue;
            ParseExpr(eValue);
            if (uArrayIndex > 0xff) {
                const std::uint8_t uValReg = Exp2AnyReg(eValue);
                ExpDesc_t eKey;
                eKey.eKind = EExpKind::KNumber;
                eKey.payload.tvValue = TValue_t::Number(static_cast<double>(uArrayIndex));
                const std::uint8_t uKeyReg = Exp2AnyReg(eKey);
                EmitABC(EBcOp::TSetV, uValReg, uTableReg, uKeyReg);
                FreeExp(eKey);
                FreeExp(eValue);
            } else {
                const std::uint8_t uValReg = Exp2AnyReg(eValue);
                EmitABC(EBcOp::TSetB, uValReg, uTableReg,
                        static_cast<std::uint8_t>(uArrayIndex));
                FreeExp(eValue);
            }
            ++uArrayIndex;
        }
        if (!Opt(static_cast<ETokenKind>(',')) && !Opt(static_cast<ETokenKind>(';'))) break;
    }
    Expect(static_cast<ETokenKind>('}'), "}");
    // Patch the size hint: low 11 bits array size, top 5 bits log2 hash size.
    std::uint32_t uHashLog = 0;
    while ((1u << uHashLog) < uHashCount) ++uHashLog;
    const std::uint16_t uHint = static_cast<std::uint16_t>(
        ((uArrayIndex < 0x800 ? uArrayIndex : 0x7ff)) | (uHashLog << 11));
    fs->vCode[uTNewPos].uRaw = (fs->vCode[uTNewPos].uRaw & 0xffff) |
                               (std::uint32_t{uHint} << 16);
}

}  // namespace

// ---------------------------------------------------------------------------
// Parsing (part 3) — statements, function assembly, public entry.
// ---------------------------------------------------------------------------

namespace {

std::uint32_t Ctx_t::ParseExprList(ExpDesc_t& eLast) {
    std::uint32_t uCount = 1;
    ParseExpr(eLast);
    while (Opt(static_cast<ETokenKind>(','))) {
        Exp2NextReg(eLast);
        ParseExpr(eLast);
        ++uCount;
    }
    return uCount;
}

void Ctx_t::AdjustAssign(std::uint32_t uWanted, std::uint32_t uGiven, ExpDesc_t& eLast) {
    std::int32_t nExtra = static_cast<std::int32_t>(uWanted) - static_cast<std::int32_t>(uGiven);
    if (eLast.eKind == EExpKind::Call) {
        if (nExtra < 0) nExtra = 0;
        // Expand the call to produce 1+extra results.
        BcIns_t& ins = fs->vCode[eLast.payload.slotPair.uInfo];
        ins.uRaw = (ins.uRaw & ~0xff000000u) |
                   (static_cast<std::uint32_t>(nExtra + 2) << 24);
        if (nExtra > 0) ReserveRegs(static_cast<std::uint32_t>(nExtra));
        fs->uFreeReg = eLast.payload.slotPair.uAux + 1 + static_cast<std::uint32_t>(nExtra);
        return;
    }
    if (eLast.eKind != EExpKind::Void) Exp2NextReg(eLast);
    if (nExtra > 0) {
        const std::uint32_t uFirst = fs->uFreeReg;
        ReserveRegs(static_cast<std::uint32_t>(nExtra));
        EmitAD(EBcOp::KNil, static_cast<std::uint8_t>(uFirst),
               static_cast<std::uint16_t>(uFirst + static_cast<std::uint32_t>(nExtra) - 1));
    } else if (nExtra < 0) {
        fs->uFreeReg += static_cast<std::uint32_t>(nExtra);  // drop extras
    }
}

void Ctx_t::ParseLocal() {
    if (Tok() == ETokenKind::Function) {  // local function f
        Next();
        C_GcString* pName = ExpectName();
        NewLocal(pName);
        ActivateLocals(1);           // visible inside its own body (recursion)
        ReserveRegs(0);
        ExpDesc_t eFunc;
        ParseFunctionBody(eFunc, false);
        Exp2Reg(eFunc, static_cast<std::uint8_t>(fs->uNumActive - 1));
        return;
    }
    std::vector<C_GcString*> vNames;
    vNames.push_back(ExpectName());
    while (Opt(static_cast<ETokenKind>(','))) vNames.push_back(ExpectName());
    ExpDesc_t eLast;
    std::uint32_t uGiven = 0;
    if (Opt(static_cast<ETokenKind>('='))) {
        uGiven = ParseExprList(eLast);
    } else {
        eLast.eKind = EExpKind::Void;
    }
    AdjustAssign(static_cast<std::uint32_t>(vNames.size()), uGiven, eLast);
    for (C_GcString* pName : vNames) NewLocal(pName);
    ActivateLocals(static_cast<std::uint32_t>(vNames.size()));
}

void Ctx_t::ParseAssignmentOrCall() {
    ExpDesc_t eFirst;
    ParseSuffixedExpr(eFirst);
    if (Tok() != static_cast<ETokenKind>('=') && Tok() != static_cast<ETokenKind>(',')) {
        if (eFirst.eKind != EExpKind::Call) Error("syntax error: statement expected");
        // Call statement: zero results.
        fs->vCode[eFirst.payload.slotPair.uInfo].uRaw =
            (fs->vCode[eFirst.payload.slotPair.uInfo].uRaw & ~0xff000000u) |
            (std::uint32_t{1} << 24);
        fs->uFreeReg = eFirst.payload.slotPair.uAux;
        return;
    }
    std::vector<ExpDesc_t> vTargets;
    vTargets.push_back(eFirst);
    while (Opt(static_cast<ETokenKind>(','))) {
        ExpDesc_t eTarget;
        ParseSuffixedExpr(eTarget);
        vTargets.push_back(eTarget);
    }
    Expect(static_cast<ETokenKind>('='), "=");
    ExpDesc_t eLast;
    const std::uint32_t uGiven = ParseExprList(eLast);
    if (vTargets.size() == 1 && uGiven == 1) {
        StoreVar(vTargets[0], eLast);
        return;
    }
    AdjustAssign(static_cast<std::uint32_t>(vTargets.size()), uGiven, eLast);
    // Values sit in consecutive registers ending at freereg; assign right→left.
    std::uint32_t uValueReg = fs->uFreeReg;
    for (std::size_t uI = vTargets.size(); uI-- > 0;) {
        --uValueReg;
        ExpDesc_t eValue;
        eValue.eKind = EExpKind::NonReloc;
        eValue.payload.slotPair.uInfo = uValueReg;
        StoreVar(vTargets[uI], eValue);
    }
    fs->uFreeReg = fs->uNumActive;
}

void Ctx_t::ParseIf() {
    // if cond then block {elseif cond then block} [else block] end
    std::uint32_t posEscapes = kNoJump;
    for (;;) {
        Next();  // skip 'if' / 'elseif'
        ExpDesc_t eCond;
        ParseExpr(eCond);
        GoIfTrue(eCond);
        Expect(ETokenKind::Then, "then");
        BlockScope_t block;
        EnterBlock(block, false);
        ParseBlock();
        LeaveBlock();
        if (Tok() == ETokenKind::ElseIf) {
            Concat(posEscapes, EmitJump());
            PatchToHere(eCond.posFalseList);
            continue;
        }
        if (Tok() == ETokenKind::Else) {
            Concat(posEscapes, EmitJump());
            PatchToHere(eCond.posFalseList);
            Next();
            BlockScope_t blockElse;
            EnterBlock(blockElse, false);
            ParseBlock();
            LeaveBlock();
        } else {
            PatchToHere(eCond.posFalseList);
        }
        break;
    }
    Expect(ETokenKind::End, "end");
    PatchToHere(posEscapes);
}

void Ctx_t::ParseWhile() {
    Next();
    const std::uint32_t uTop = Pos();
    ExpDesc_t eCond;
    ParseExpr(eCond);
    GoIfTrue(eCond);
    Expect(ETokenKind::Do, "do");
    BlockScope_t block;
    EnterBlock(block, true);
    EmitAD(EBcOp::Loop, static_cast<std::uint8_t>(fs->uNumActive), vm::kJumpBias);
    ParseBlock();
    FixJump(EmitJump(), uTop);
    Expect(ETokenKind::End, "end");
    LeaveBlock();
    PatchToHere(eCond.posFalseList);
}

void Ctx_t::ParseRepeat() {
    Next();
    const std::uint32_t uTop = Pos();
    BlockScope_t block;
    EnterBlock(block, true);
    EmitAD(EBcOp::Loop, static_cast<std::uint8_t>(fs->uNumActive), vm::kJumpBias);
    // Body scope stays open across 'until' (its locals are visible there).
    BlockScope_t blockBody;
    EnterBlock(blockBody, false);
    ParseBlock();
    Expect(ETokenKind::Until, "until");
    ExpDesc_t eCond;
    ParseExpr(eCond);
    // until cond: fall through (exit) when true, jump back to top when false.
    // GoIfTrue leaves the false-list as the "condition was false" jumps.
    GoIfTrue(eCond);
    PatchTo(eCond.posFalseList, uTop);
    eCond.posFalseList = kNoJump;
    LeaveBlock();  // body locals die here
    LeaveBlock();
}

void Ctx_t::ParseFor() {
    Next();
    C_GcString* pFirstName = ExpectName();
    if (Tok() == static_cast<ETokenKind>('=')) {
        // Numeric for: idx/stop/step control slots + visible ext slot.
        Next();
        const std::uint32_t uBase = fs->uFreeReg;
        ExpDesc_t eExpr;
        ParseExpr(eExpr);
        Exp2NextReg(eExpr);   // start
        Expect(static_cast<ETokenKind>(','), ",");
        ParseExpr(eExpr);
        Exp2NextReg(eExpr);   // stop
        if (Opt(static_cast<ETokenKind>(','))) {
            ParseExpr(eExpr);
            Exp2NextReg(eExpr);  // step
        } else {
            EmitAD(EBcOp::KShort, static_cast<std::uint8_t>(fs->uFreeReg), 1);
            ReserveRegs(1);
        }
        // Hidden control locals + the visible variable.
        NewLocal(uni.Interner().Intern("(for index)"));
        NewLocal(uni.Interner().Intern("(for limit)"));
        NewLocal(uni.Interner().Intern("(for step)"));
        ActivateLocals(3);
        Expect(ETokenKind::Do, "do");
        const std::uint32_t posForI = EmitAD(EBcOp::ForI, static_cast<std::uint8_t>(uBase), 0);
        BlockScope_t block;
        EnterBlock(block, true);
        NewLocal(pFirstName);
        ActivateLocals(1);   // ext slot = uBase+3
        ReserveRegs(0);
        ParseBlock();
        Expect(ETokenKind::End, "end");
        const std::uint32_t posForL = EmitAD(EBcOp::ForL, static_cast<std::uint8_t>(uBase), 0);
        FixJump(posForL, posForI + 1);
        FixJump(posForI, Pos());
        LeaveBlock();
        fs->uNumActive -= 3;  // drop hidden control locals
        fs->vLocals.resize(fs->uNumActive);
        fs->uFreeReg = fs->uNumActive;
        return;
    }
    // Generic for: for a[,b...] in explist do ... end
    std::vector<C_GcString*> vNames;
    vNames.push_back(pFirstName);
    while (Opt(static_cast<ETokenKind>(','))) vNames.push_back(ExpectName());
    Expect(ETokenKind::In, "in");
    const std::uint32_t uBase = fs->uFreeReg;   // func/state/ctrl land here
    ExpDesc_t eLast;
    const std::uint32_t uGiven = ParseExprList(eLast);
    AdjustAssign(3, uGiven, eLast);
    NewLocal(uni.Interner().Intern("(for generator)"));
    NewLocal(uni.Interner().Intern("(for state)"));
    NewLocal(uni.Interner().Intern("(for control)"));
    ActivateLocals(3);
    Expect(ETokenKind::Do, "do");
    const std::uint32_t posPrep = EmitJump();
    BlockScope_t block;
    EnterBlock(block, true);
    for (C_GcString* pName : vNames) NewLocal(pName);
    ActivateLocals(static_cast<std::uint32_t>(vNames.size()));
    ReserveRegs(0);
    const std::uint32_t uLoopTop = Pos();
    EmitAD(EBcOp::Loop, static_cast<std::uint8_t>(fs->uNumActive), vm::kJumpBias);
    ParseBlock();
    Expect(ETokenKind::End, "end");
    FixJump(posPrep, Pos());
    // Iterator step: R[base+3..] = R[base](R[base+1], R[base+2])
    EmitABC(EBcOp::IterC, static_cast<std::uint8_t>(uBase + 3),
            static_cast<std::uint8_t>(vNames.size() + 1), 3);
    const std::uint32_t posIterL =
        EmitAD(EBcOp::IterL, static_cast<std::uint8_t>(uBase + 3), 0);
    FixJump(posIterL, uLoopTop);
    LeaveBlock();
    fs->uNumActive -= 3;
    fs->vLocals.resize(fs->uNumActive);
    fs->uFreeReg = fs->uNumActive;
}

void Ctx_t::ParseReturn() {
    Next();
    // return [explist]
    if (Tok() == ETokenKind::End || Tok() == ETokenKind::Eof ||
        Tok() == ETokenKind::Else || Tok() == ETokenKind::ElseIf ||
        Tok() == ETokenKind::Until || Tok() == static_cast<ETokenKind>(';')) {
        EmitCloseIfCaptured(0);
        EmitAD(EBcOp::Ret0, 0, 1);
    } else {
        const std::uint32_t uFirst = fs->uFreeReg;
        ExpDesc_t eLast;
        const std::uint32_t uCount = ParseExprList(eLast);
        if (eLast.eKind == EExpKind::Call && uCount == 1 && !fs->bAnyCaptured) {
            // Tailcall: rewrite the call instruction in place.
            BcIns_t& ins = fs->vCode[eLast.payload.slotPair.uInfo];
            const EBcOp eCallOp = ins.Op();
            const std::uint8_t uCallBase = ins.A();
            const std::uint8_t uArgsC = ins.C();
            ins = BcIns_t::MakeAD(eCallOp == EBcOp::CallM ? EBcOp::CallMT : EBcOp::CallT,
                                  uCallBase, uArgsC);
        } else if (eLast.eKind == EExpKind::Call) {
            // Multi-result tail: return all results of the last call.
            fs->vCode[eLast.payload.slotPair.uInfo].uRaw &= ~0xff000000u;  // B=0
            EmitCloseIfCaptured(0);
            EmitAD(EBcOp::RetM, static_cast<std::uint8_t>(uFirst),
                   static_cast<std::uint16_t>(uCount - 1));
        } else {
            Exp2NextReg(eLast);
            EmitCloseIfCaptured(0);
            if (uCount == 1) {
                EmitAD(EBcOp::Ret1, static_cast<std::uint8_t>(uFirst), 2);
            } else {
                EmitAD(EBcOp::Ret, static_cast<std::uint8_t>(uFirst),
                       static_cast<std::uint16_t>(uCount + 1));
            }
        }
        fs->uFreeReg = uFirst;
    }
    (void)Opt(static_cast<ETokenKind>(';'));
}

void Ctx_t::ParseFunctionStatement() {
    Next();
    // funcname: Name {'.' Name} [':' Name]
    ExpDesc_t eTarget;
    SingleVar(eTarget, ExpectName());
    bool bIsMethod = false;
    while (Tok() == static_cast<ETokenKind>('.') || Tok() == static_cast<ETokenKind>(':')) {
        const bool bColon = Tok() == static_cast<ETokenKind>(':');
        Next();
        C_GcString* pField = ExpectName();
        ExpDesc_t eKey;
        eKey.eKind = EExpKind::KString;
        eKey.payload.tvValue = TValue_t::GcObject(vm::EValueTag::String, pField);
        IndexedKey(eTarget, eKey);
        if (bColon) {
            bIsMethod = true;
            break;
        }
    }
    ExpDesc_t eFunc;
    ParseFunctionBody(eFunc, bIsMethod);
    StoreVar(eTarget, eFunc);
}

bool Ctx_t::ParseStatement() {
    switch (static_cast<std::uint16_t>(Tok())) {
        case ';': Next(); return false;
        case static_cast<std::uint16_t>(ETokenKind::If): ParseIf(); return false;
        case static_cast<std::uint16_t>(ETokenKind::While): ParseWhile(); return false;
        case static_cast<std::uint16_t>(ETokenKind::Repeat): ParseRepeat(); return false;
        case static_cast<std::uint16_t>(ETokenKind::For): ParseFor(); return false;
        case static_cast<std::uint16_t>(ETokenKind::Do): {
            Next();
            BlockScope_t block;
            EnterBlock(block, false);
            ParseBlock();
            LeaveBlock();
            Expect(ETokenKind::End, "end");
            return false;
        }
        case static_cast<std::uint16_t>(ETokenKind::Local): Next(); ParseLocal(); return false;
        case static_cast<std::uint16_t>(ETokenKind::Function): ParseFunctionStatement(); return false;
        case static_cast<std::uint16_t>(ETokenKind::Return): ParseReturn(); return true;
        case static_cast<std::uint16_t>(ETokenKind::Break): {
            Next();
            BlockScope_t* pLoop = fs->pBlock;
            while (pLoop && !pLoop->bIsLoop) pLoop = pLoop->pPrev;
            if (!pLoop) Error("'break' outside a loop");
            EmitCloseIfCaptured(pLoop->uFirstLocal);
            Concat(pLoop->posBreakList, EmitJump());
            return true;
        }
        case static_cast<std::uint16_t>(ETokenKind::Goto): Error("'goto' is not supported yet");
        default: ParseAssignmentOrCall(); return false;
    }
}

void Ctx_t::ParseBlock() {
    for (;;) {
        switch (static_cast<std::uint16_t>(Tok())) {
            case static_cast<std::uint16_t>(ETokenKind::Eof):
            case static_cast<std::uint16_t>(ETokenKind::End):
            case static_cast<std::uint16_t>(ETokenKind::Else):
            case static_cast<std::uint16_t>(ETokenKind::ElseIf):
            case static_cast<std::uint16_t>(ETokenKind::Until):
                return;
            default: break;
        }
        const bool bTerminator = ParseStatement();
        if (fs->uFreeReg < fs->uNumActive) Error("internal: register leak");
        fs->uFreeReg = fs->uNumActive;
        if (bTerminator) return;
    }
}

C_GcProto* Ctx_t::FinishFunction() {
    // Implicit final return.
    EmitCloseIfCaptured(0);
    EmitAD(EBcOp::Ret0, 0, 1);

    // Layout: [proto][bc][kgc (reversed, grows down)][knum][uvdescs]
    const std::uint32_t uBcCount = static_cast<std::uint32_t>(fs->vCode.size());
    const std::uint32_t uKgc = static_cast<std::uint32_t>(fs->vGc.size());
    const std::uint32_t uKnum = static_cast<std::uint32_t>(fs->vNum.size());
    const std::uint32_t uUpvals = static_cast<std::uint32_t>(fs->vUpvals.size());
    std::size_t uOfs = sizeof(C_GcProto) + uBcCount * 4;
    uOfs = (uOfs + uKgc * 4 + 7) & ~std::size_t{7};  // kgc block then align knum to 8
    const std::size_t uKnumOfs = uOfs;
    uOfs += uKnum * 8;
    const std::size_t uUvOfs = uOfs;
    uOfs += uUpvals * 2;
    uOfs = (uOfs + 7) & ~std::size_t{7};

    auto* pProto = static_cast<C_GcProto*>(
        uni.Gc().AllocObject(vm::EGcObjectType::Proto, uOfs));
    pProto->m_Header.uExtra1 = fs->uNumParams;
    pProto->m_Header.uExtra2 = static_cast<std::uint8_t>(fs->uFrameMax);
    pProto->m_uBcCount = uBcCount;
    pProto->m_uGcConstCount = uKgc;
    pProto->m_uNumConstCount = uKnum;
    pProto->m_uTotalSize = static_cast<std::uint32_t>(uOfs);
    pProto->m_uUpvalCount = static_cast<std::uint8_t>(uUpvals);
    pProto->m_uFlags = 0;
    pProto->m_uRootTrace = 0;
    pProto->m_rChunkName = uni.MakeRef(lex.ChunkName());
    pProto->m_rGcList = core::GcRef_t{};

    auto* pBc = const_cast<std::uint32_t*>(pProto->Bytecode());
    for (std::uint32_t uI = 0; uI < uBcCount; ++uI) pBc[uI] = fs->vCode[uI].uRaw;

    auto* pBase = reinterpret_cast<char*>(pProto);
    auto* pKnum = reinterpret_cast<TValue_t*>(pBase + uKnumOfs);
    pProto->m_rConstants = core::PtrToRef(uni.ArenaBase(), pKnum);
    for (std::uint32_t uI = 0; uI < uKnum; ++uI) pKnum[uI] = TValue_t::Number(fs->vNum[uI]);
    auto* pKgcArea = reinterpret_cast<core::GcRef_t*>(pKnum);
    for (std::uint32_t uI = 0; uI < uKgc; ++uI)
        pKgcArea[-static_cast<std::int32_t>(uI + 1)] =
            uni.MakeRef(fs->vGc[uI].AsGcPointer());

    auto* pUvDescs = reinterpret_cast<std::uint16_t*>(pBase + uUvOfs);
    pProto->m_rUpvalDescs = core::PtrToRef(uni.ArenaBase(), pUvDescs);
    for (std::uint32_t uI = 0; uI < uUpvals; ++uI) {
        const UpvalDesc_t& desc = fs->vUpvals[uI];
        pUvDescs[uI] = static_cast<std::uint16_t>(
            desc.bFromParentLocal ? (0x8000u | desc.uIndex) : desc.uIndex);
    }

    // Line info as a side vector.
    auto* pLines = static_cast<core::BcLine_t*>(
        uni.Allocator().AllocVector(uBcCount * sizeof(core::BcLine_t)));
    for (std::uint32_t uI = 0; uI < uBcCount; ++uI) pLines[uI] = fs->vLines[uI];
    pProto->m_rLineInfo = core::PtrToRef(uni.ArenaBase(), pLines);
    return pProto;
}

void Ctx_t::ParseFunctionBody(ExpDesc_t& e, bool bIsMethod) {
    FuncState_t funcState;
    funcState.pParent = fs;
    fs = &funcState;

    // Header placeholder (patched by FinishFunction's framesize).
    EmitAD(EBcOp::FuncF, 0, 0);

    Expect(static_cast<ETokenKind>('('), "(");
    if (bIsMethod) {
        NewLocal(uni.Interner().Intern("self"));
        ActivateLocals(1);
        ++funcState.uNumParams;
    }
    if (Tok() != static_cast<ETokenKind>(')')) {
        for (;;) {
            if (Tok() == ETokenKind::Ellipsis) Error("'...' is not supported yet");
            NewLocal(ExpectName());
            ActivateLocals(1);
            ++funcState.uNumParams;
            if (!Opt(static_cast<ETokenKind>(','))) break;
        }
    }
    Expect(static_cast<ETokenKind>(')'), ")");
    funcState.uFreeReg = funcState.uNumActive;

    ParseBlock();
    Expect(ETokenKind::End, "end");
    C_GcProto* pProto = FinishFunction();
    // Patch the header with the final frame size.
    auto* pBc = const_cast<std::uint32_t*>(pProto->Bytecode());
    pBc[0] = BcIns_t::MakeAD(EBcOp::FuncF, pProto->m_Header.uExtra2, 0).uRaw;

    fs = funcState.pParent;

    // Closure creation in the ENCLOSING function.
    const std::uint32_t uIdx =
        GcConst(TValue_t::GcObject(vm::EValueTag::Proto, pProto));
    e = ExpDesc_t{};
    e.payload.slotPair.uInfo = EmitAD(EBcOp::FNew, 0, static_cast<std::uint16_t>(uIdx));
    e.eKind = EExpKind::Relocatable;
}

}  // namespace

vm::C_GcProto* C_Parser::ParseChunk() {
    Ctx_t ctx{*m_pLexer, *m_pUniverse, nullptr};
    FuncState_t chunkState;
    ctx.fs = &chunkState;
    ctx.EmitAD(EBcOp::FuncF, 0, 0);
    ctx.ParseBlock();
    if (ctx.Tok() != ETokenKind::Eof) ctx.Error("'<eof>' expected");
    C_GcProto* pProto = ctx.FinishFunction();
    auto* pBc = const_cast<std::uint32_t*>(pProto->Bytecode());
    pBc[0] = BcIns_t::MakeAD(EBcOp::FuncF, pProto->m_Header.uExtra2, 0).uRaw;
    return pProto;
}

}  // namespace ljx::fe
