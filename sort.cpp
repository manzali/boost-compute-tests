#include <vector>
#include <algorithm>
#include <boost/compute.hpp>

namespace compute = boost::compute;

BOOST_COMPUTE_FUNCTION(int, add_three, (int x),
{
 return x + 3;
});

int main()
{
	size_t const dim = 100000000;
	
    // generate random numbers on the host
    std::vector<int> host_vector_a(dim);
    std::fill(host_vector_a.begin(), host_vector_a.end(), 999);
	
    std::vector<int> host_vector_b(dim);

    std::vector<compute::platform> platforms = compute::system::platforms();

    for(size_t i = 0; i < platforms.size(); i++){
        const compute::platform &platform = platforms[i];

        std::cout << "Platform '" << platform.name() << "'\n\n";

        std::vector<compute::device> devices = platform.devices();
        for(size_t j = 0; j < devices.size(); j++){
            compute::device const& dev = devices[j];

            std::string type;
            if(dev.type() & compute::device::gpu)
                type = "GPU Device";
            else if(dev.type() & compute::device::cpu)
                type = "CPU Device";
            else if(dev.type() & compute::device::accelerator)
                type = "Accelerator Device";
            else
                type = "Unknown Device";

            std::cout << "  " << type << ":\n";
			std::cout << "    Name: " << dev.name() << std::endl;
			std::cout << "    Vendor: " << dev.vendor() << std::endl;
			std::cout << "    Device version: " << dev.version() << std::endl;
			std::cout << "    Driver version: " << dev.driver_version() << std::endl;
			/*
			std::cout << "    Supported extensions:\n";
			std::vector<std::string> extensions = dev.extensions();
			for(auto const& e : extensions){
				if(!e.empty()){
					std::cout << "      " << e << std::endl;
				}
			}
			*/
			std::cout << "    Address bits: " << dev.address_bits() << std::endl;
			std::cout << "    Global memory size in bytes: " << dev.global_memory_size() << std::endl;
			std::cout << "    Local memory size in bytes: " << dev.local_memory_size() << std::endl;
			std::cout << "    Clock frequency in Hertz: " << dev.clock_frequency() << std::endl;
			std::cout << "    Compute units: " << dev.compute_units() << std::endl;
			std::cout << "    Max work group size: " << dev.max_work_group_size() << std::endl;
			std::cout << "    Profiling timer resolution in nanoseconds: " << dev.profiling_timer_resolution() << std::endl;

		    // create a compute context and command queue
		    compute::context ctx(dev);
		    compute::command_queue queue(ctx, dev, compute::command_queue::enable_profiling);

		    // create vector on the device
		    compute::vector<float> device_vector(dim, ctx);

		    // copy data to the device
		    auto future_1 = compute::copy_async(
		        host_vector_a.begin(), host_vector_a.end(), device_vector.begin(), queue
		    );

		    // wait for copy to finish
		    future_1.wait();

		    // print elapsed time in microseconds
		    std::cout << "\n    Copy to device: " << future_1.get_event().duration<boost::chrono::microseconds>().count() << " us\n";
	
		    // sort data on the device
			auto start = std::chrono::high_resolution_clock::now();
		    compute::transform(
		        device_vector.begin(), device_vector.end(), device_vector.begin(), add_three, queue
		    );
			auto end = std::chrono::high_resolution_clock::now();
			auto us = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
			std::cout << "    Computation: " << us.count() << " us\n";

		    // copy data back to the host
		    auto future_2 = compute::copy_async(
		        device_vector.begin(), device_vector.end(), host_vector_b.begin(), queue
		    );

		    // wait for copy to finish
		    future_2.wait();

		    // print elapsed time in microseconds
		    std::cout << "    Copy from device: " << future_2.get_event().duration<boost::chrono::microseconds>().count() << " us\n\n";
		
        }		
    }	

    return 0;
}