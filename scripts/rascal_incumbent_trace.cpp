// Observe RDKit 2024.03.5 accepted incumbent updates without changing search.
// The exported symbol and signature are checked against the pinned wheel.
#include <dlfcn.h>
#include <fcntl.h>
#include <time.h>
#include <unistd.h>
#include <cstdlib>
#include <cstdio>
#include <sstream>
#include <utility>
#include <vector>

namespace RDKit {
class ROMol;
namespace RascalMCES {
struct RascalOptions;
using Clique = std::vector<unsigned int>;
using Cliques = std::vector<Clique>;
using Pairs = std::vector<std::pair<int, int>>;

void updateMaxClique(const Clique &clique, bool deltaYPoss,
                     const RascalOptions &opts, const ROMol &mol1,
                     const ROMol &mol2, const Pairs &vtxPairs,
                     Cliques &maxCliques, unsigned int &lowerBound) {
  using Function = void (*)(const Clique &, bool, const RascalOptions &,
                           const ROMol &, const ROMol &, const Pairs &,
                           Cliques &, unsigned int &);
  static Function original = []() {
    const char *path = std::getenv("NGA_RASCAL_LIBRARY");
    void *handle = path ? dlopen(path, RTLD_LAZY | RTLD_NOLOAD) : nullptr;
    const char *symbol = "_ZN5RDKit10RascalMCES15updateMaxCliqueERKSt6vectorIjSaIjEEbRKNS0_13RascalOptionsERKNS_5ROMolESB_RKS1_ISt4pairIiiESaISD_EERS1_IS3_SaIS3_EERj";
    auto fn = handle ? reinterpret_cast<Function>(dlsym(handle, symbol)) : nullptr;
    if (!fn || fn == &updateMaxClique) {
      std::fprintf(stderr, "Incumbent observer could not resolve original pinned RDKit symbol\n");
      std::_Exit(91);
    }
    return fn;
  }();
  original(clique, deltaYPoss, opts, mol1, mol2, vtxPairs, maxCliques, lowerBound);
  static size_t emitted = 0;
  if (maxCliques.empty() || maxCliques.front().size() <= emitted) return;
  const char *path = std::getenv("NGA_RASCAL_TRACE");
  if (!path) return;
  static int fd = open(path, O_CREAT | O_WRONLY | O_APPEND, 0600);
  if (fd < 0) std::_Exit(92);
  timespec now{};
  clock_gettime(CLOCK_MONOTONIC, &now);
  std::ostringstream out;
  out << "{\"monotonic_ns\":" << (static_cast<long long>(now.tv_sec)*1000000000LL+now.tv_nsec)
      << ",\"bond_pairs\":[";
  bool first = true;
  for (auto index : maxCliques.front()) {
    if (!first) out << ',';
    first = false;
    out << '[' << vtxPairs[index].first << ',' << vtxPairs[index].second << ']';
  }
  out << "]}\n";
  const auto value = out.str();
  size_t offset = 0;
  while (offset < value.size()) {
    auto n = write(fd, value.data()+offset, value.size()-offset);
    if (n <= 0) std::_Exit(93);
    offset += static_cast<size_t>(n);
  }
  emitted = maxCliques.front().size();
}
}  // namespace RascalMCES
}  // namespace RDKit
