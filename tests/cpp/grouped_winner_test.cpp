#include <cassert>
#include <set>

#include "hnswlib/hnswlib.h"

int main() {
    hnswlib::L2Space space(1);
    hnswlib::HierarchicalNSW<float> index(&space, 3, 2, 10);

    const float tied = 1.0f;
    const float other = 3.0f;
    index.addPoint(&tied, 1);
    index.addPoint(&tied, 0);
    index.addPoint(&other, 2);

    const uint32_t groups[] = {0, 0, 1};
    index.setGroupMap(groups);
    index.setEf(10);

    const float query = 0.0f;
    const auto results = index.searchKnn(&query, 2);
    assert(results.size() == 2);

    std::set<hnswlib::labeltype> labels;
    for (const auto &result : results)
        labels.insert(result.second);

    // Group 0 has an exact tie: the lower global vector label must win.
    assert(labels == std::set<hnswlib::labeltype>({0, 2}));

    // Exercise the full result-list boundary too: the equal-distance lower label
    // must still be admitted when ef is already full.
    hnswlib::HierarchicalNSW<float> tight(&space, 2, 2, 10);
    tight.addPoint(&tied, 1);
    tight.addPoint(&tied, 0);
    const uint32_t one_group[] = {0, 0};
    tight.setGroupMap(one_group);
    tight.setEf(1);
    const auto tied_result = tight.searchKnn(&query, 1);
    assert(tied_result.size() == 1);
    assert(tied_result.front().second == 0);
    return 0;
}
