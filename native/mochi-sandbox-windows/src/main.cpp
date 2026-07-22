#include <iostream>
#include <string>

namespace {

constexpr int kProtocolVersion = 1;

bool IsBoundedNonce(const std::string& nonce) {
  if (nonce.size() < 16 || nonce.size() > 128) {
    return false;
  }
  for (const char character : nonce) {
    const bool safe = (character >= 'a' && character <= 'z') ||
                      (character >= 'A' && character <= 'Z') ||
                      (character >= '0' && character <= '9') || character == '-' ||
                      character == '_';
    if (!safe) {
      return false;
    }
  }
  return true;
}

int EmitUnavailableHello(const std::string& nonce) {
  std::cout
      << "{\"nonce\":\"" << nonce
      << "\",\"payload\":{\"backend\":\"windows-appcontainer\","
         "\"capabilities\":{\"available\":false,\"backend\":"
         "\"windows-appcontainer\",\"degraded_reason\":"
         "\"windows_native_containment_not_implemented\",\"detached\":false,"
         "\"filesystem\":false,\"last_probe_at\":null,\"network\":false,"
         "\"process\":false,\"version\":\"scaffold-1\"},"
         "\"version\":\"scaffold-1\"},\"protocol_version\":"
      << kProtocolVersion << ",\"type\":\"hello\"}\n";
  return 0;
}

}  // namespace

int main(int argc, char* argv[]) {
  if (argc == 4 && std::string(argv[1]) == "--probe" &&
      std::string(argv[2]) == "--nonce") {
    const std::string nonce(argv[3]);
    if (!IsBoundedNonce(nonce)) {
      std::cerr << "invalid nonce\n";
      return 64;
    }
    return EmitUnavailableHello(nonce);
  }

  // Run/cancel intentionally remain unavailable. Advertising containment before
  // AppContainer, ACL journal ownership, Job Object, and network tests pass would
  // turn a defense-in-depth feature into a sandbox bypass.
  std::cerr << "sandbox execution unavailable: native containment is not implemented\n";
  return 78;
}
