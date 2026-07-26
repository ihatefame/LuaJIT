// LJX — bytecode opcode metadata (names; op-info table stub for v1).
#include "ljx/vm/Bytecode.hpp"

namespace ljx::vm {

namespace {
constexpr const char* kNames[] = {
#define LJX_BC_NAME(name) #name,
    LJX_BC_REGISTRY(LJX_BC_NAME)
#undef LJX_BC_NAME
};
}  // namespace

const char* OpName(EBcOp eOp) noexcept {
    const auto uIdx = static_cast<std::uint32_t>(eOp);
    return uIdx < static_cast<std::uint32_t>(EBcOp::Count_) ? kNames[uIdx] : "?builtin";
}

const BcOpInfo_t& OpInfo(EBcOp) noexcept {
    static const BcOpInfo_t kStub{};
    return kStub;
}

}  // namespace ljx::vm
