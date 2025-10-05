// This legacy entry point has been replaced by compare_build and compare_search.
#include <cstdlib>
#include <iostream>

int main()
{
	std::cerr << "compare_single_and_distribute_compute_cost has been superseded.\n"
			  << "Use compare_build to construct shard manifests and compare_search to evaluate them." << std::endl;
	return EXIT_FAILURE;
}
