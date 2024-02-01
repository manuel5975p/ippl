#include "Types/Variant.h"
#include <memory>
struct A{};
int main(){
    ippl::Variant<int, std::unique_ptr<int>> v(std::make_unique<int>(5));
    ippl::Variant<int, std::unique_ptr<int>> v2(std::move(v));
    std::cout << *(v2.get<1>()) << "\n";
    //for(int i = 0;i < 1000;i++){
    //    v = std::make_unique<int>(i);
    //    std::cout << *(v.get<1>()) << "\n";
    //}
    //std::variant<int, float>(3.0f);
}