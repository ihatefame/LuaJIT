// LJX — command-line runner: ljx <script.lua>
#include <csetjmp>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#include "ljx/fe/Lexer.hpp"
#include "ljx/fe/Parser.hpp"
#include "ljx/gc/GarbageCollector.hpp"
#include "ljx/rt/StringInterner.hpp"
#include "ljx/vm/Interpreter.hpp"

namespace {

struct FileReader_t {
    const char* pData;
    std::size_t uSize;
    bool bDone;
};

const char* ReadAll(void* pUserData, std::size_t* puSize) {
    auto* pReader = static_cast<FileReader_t*>(pUserData);
    if (pReader->bDone) {
        *puSize = 0;
        return nullptr;
    }
    pReader->bDone = true;
    *puSize = pReader->uSize;
    return pReader->pData;
}

}  // namespace

int main(int nArgc, char** vArgv) {
    if (nArgc < 2) {
        std::fprintf(stderr, "usage: ljx <script.lua>\n");
        return 2;
    }
    std::FILE* pFile = std::fopen(vArgv[1], "rb");
    if (!pFile) {
        std::fprintf(stderr, "ljx: cannot open %s\n", vArgv[1]);
        return 2;
    }
    std::fseek(pFile, 0, SEEK_END);
    const long nSize = std::ftell(pFile);
    std::fseek(pFile, 0, SEEK_SET);
    std::string sSource(static_cast<std::size_t>(nSize), '\0');
    if (std::fread(sSource.data(), 1, sSource.size(), pFile) != sSource.size()) {
        std::fprintf(stderr, "ljx: read error\n");
        return 2;
    }
    std::fclose(pFile);

    using namespace ljx;
    vm::C_Universe* pUni = vm::C_Universe::Create();
    if (!pUni) {
        std::fprintf(stderr, "ljx: cannot create VM (no entropy or address space?)\n");
        return 2;
    }
    vm::OpenStdLib(*pUni);

    int nStatus = 0;
    {
        vm::C_GcString* pChunkName = pUni->Interner().Intern(vArgv[1]);
        pUni->Gc().FixObject(&pChunkName->m_Header);
        FileReader_t reader{sSource.data(), sSource.size(), false};
        fe::C_Lexer lexer(*pUni, &ReadAll, &reader, pChunkName);
        fe::C_Parser parser(lexer, *pUni);
        vm::C_GcProto* pProto = parser.ParseChunk();

        // Wrap in a closure and call with 0 arguments.
        auto* pMain = static_cast<vm::C_GcFunction*>(pUni->Gc().AllocObject(
            vm::EGcObjectType::Function, vm::C_GcFunction::LuaAllocSize(0)));
        pMain->m_Header.uExtra1 = 0;
        pMain->m_Header.uExtra2 = 0;
        pMain->m_rEnv = pUni->MakeRef(pUni->Globals());
        pMain->m_pPc = pProto->Bytecode();

        vm::C_LuaThread* pThread = pUni->MainThread();
        vm::TValue_t* pFunc = pThread->m_pTop;
        pFunc[0] = vm::TValue_t::GcObject(vm::EValueTag::Function, pMain);
        if (vm::C_Interpreter::ProtectedCall(pThread, pFunc, 0, 0, 0) < 0) {
            char vMessage[512] = "?";
            const vm::TValue_t tvErr = pFunc[0];
            if (tvErr.Is(vm::EValueTag::String)) {
                auto* pStr = static_cast<vm::C_GcString*>(tvErr.AsGcPointer());
                std::snprintf(vMessage, sizeof vMessage, "%.*s",
                              static_cast<int>(pStr->Length()), pStr->Data());
            }
            std::fprintf(stderr, "ljx: %s\n", vMessage);
            nStatus = 1;
        }
    }
    pUni->Destroy();
    return nStatus;
}
