#include "pch.h"
#include <minhook.h>
#include <cstdio>
#include <iostream>
#include <iphlpapi.h>
#include <cstring>
#include <string>
#include <functional>
#include "il2cpp.h"
// Definitions and declarations
uintptr_t base = (uintptr_t)GetModuleHandle(NULL);
uintptr_t GameAssembly = (uintptr_t)GetModuleHandle(L"GameAssembly.dll");
uintptr_t GameObject__SetActive = 0x20DDEC0;  // Actual offset for SetActive

const char* wordList[] = { "Lighthouse", "Fish", "CityRenderer", "Terrain", "ResourceSprite", "Animals", "Farm", "Road", "Port", "ImprovementSprite", "Starfish","SpriteContainer","city","plot","ProgressBar" };
const char* fog = "FogOfWar";

typedef System_String_o* (*ToString_t)(UnityEngine_GameObject_o* __this, const MethodInfo* method);
void(__cdecl* GameObject_SetActive_o)(UnityEngine_GameObject_o*, bool,DWORD*);
void __cdecl GameObject_SetActive_hook(UnityEngine_GameObject_o* __this, bool value, DWORD* method) {


    const size_t wordCount = sizeof(wordList) / sizeof(wordList[0]);
    ToString_t toStringFunc = (ToString_t)__this->klass->vtable._3_ToString.methodPtr;
    System_String_o* stringResult = toStringFunc(__this, __this->klass->vtable._3_ToString.method);

    int32_t length = stringResult->fields._stringLength;
    const uint16_t* utf16Chars = &(stringResult->fields._firstChar);

    // Convert UTF-16 to UTF-8
    static char utf8Buffer[1024];
    size_t convertedChars = 0;
    errno_t err = wcstombs_s(&convertedChars, utf8Buffer, sizeof(utf8Buffer), (const wchar_t*)utf16Chars, length);

    // Check if the conversion succeeded
    if (err != 0) {
        printf("Failed to convert GameObject ToString result to UTF-8\n");
        return;
    }

    //printf("GameObject ToString result: %s\n", utf8Buffer);

    // Check if any word in the list is in the utf8Buffer
    for (size_t i = 0; i < wordCount; ++i) {
        if (strstr(utf8Buffer, wordList[i]) != NULL) {
            // If a word is found, force value to true
            printf("Match found: %s. Forcing value to true.\n", wordList[i]);
            value = true;
            break;  // Exit loop as we've found a match
        }
        if (strstr(utf8Buffer, fog) != NULL) {
            // If a word is found, force value to true
            printf("Match found: %s. Forcing value to false.\n", fog);
            value = false;
            break;  // Exit loop as we've found a match
        }
    }

    // Call the original GameObject_SetActive function with the (possibly modified) value
    GameObject_SetActive_o(__this, value, method);
}


void CreateConsole() {
    AllocConsole();
    FILE* f;
    freopen_s(&f, "CONOUT$", "w", stdout);  // Redirect stdout to console
    freopen_s(&f, "CONIN$", "r", stdin);    // Redirect stdin to console
    std::cout.sync_with_stdio(true);        // Sync C++ and C I/O streams
}

void init() {
    CreateConsole();
    MH_Initialize();
    printf("on\n");

    MH_CreateHook(reinterpret_cast<LPVOID*>(GameAssembly + GameObject__SetActive), (LPVOID)GameObject_SetActive_hook, reinterpret_cast<LPVOID*>(&GameObject_SetActive_o));
    MH_EnableHook(reinterpret_cast<LPVOID*>(GameAssembly + GameObject__SetActive));
}


void main() {
    init();
}

void askForKey() {
    printf("You can buy a software key at Discord.gg/ if you don't have one\n");
    std::string input;
    std::cout << "Please enter your software key: ";
    std::cin >> input;
    std::cout << "Your key is: " << input << std::endl;
}
const char* GetVolumeSerialNumber() {
    static char serialStr[9];  // Static buffer to store the serial number
    DWORD volumeSerialNumber = 0;
    if (GetVolumeInformationA("C:\\", NULL, 0, &volumeSerialNumber, NULL, NULL, NULL, 0)) {
        snprintf(serialStr, sizeof(serialStr), "%08X", volumeSerialNumber);
        return serialStr;
    }
    return "None";
}
unsigned long hash_function(const char* str)
{
    unsigned long long i = 0;
    unsigned long long prime = 31;  // Use a prime multiplier for better hash distribution.

    for (int j = 0; str[j]; j++) {
        i = i * prime + str[j];  // Increase the complexity by multiplying and adding.
    }

    return i % 10000000000000007;  // Use a much larger modulus to generate a bigger hash.
}


BOOL APIENTRY DllMain(HMODULE hModule, DWORD ul_reason_for_call, LPVOID lpReserved) {
    switch (ul_reason_for_call) {
    case DLL_PROCESS_ATTACH:
        CreateThread(0, 0, (LPTHREAD_START_ROUTINE)main, 0, 0, 0);
        break;
    case DLL_THREAD_ATTACH:
    case DLL_THREAD_DETACH:
    case DLL_PROCESS_DETACH:
        break;
    }
    return TRUE;
}
