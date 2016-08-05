#include <vector>

#include <boost/compute.hpp>

namespace compute = boost::compute;

int main()
{

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
			std::cout << "    Supported extensions:\n";
			std::vector<std::string> extensions = dev.extensions();
			for(auto const& e : extensions){
				if(!e.empty()){
					std::cout << "      " << e << std::endl;
				}
			}
			std::cout << "    Address bits: " << dev.address_bits() << std::endl;
			std::cout << "    Global memory size in bytes: " << dev.global_memory_size() << std::endl;
			std::cout << "    Local memory size in bytes: " << dev.local_memory_size() << std::endl;
			std::cout << "    Clock frequency in Hertz: " << dev.clock_frequency() << std::endl;
			std::cout << "    Compute units: " << dev.compute_units() << std::endl;
			std::cout << "    Max work group size: " << dev.max_work_group_size() << std::endl;
			std::cout << "    Profiling timer resolution in nanoseconds: " << dev.profiling_timer_resolution() << "\n\n";
        }		
    }	

    return 0;
}