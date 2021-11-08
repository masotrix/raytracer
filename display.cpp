#include <boost/gil.hpp>
#include <boost/gil/extension/io/png.hpp>
#include <display.h>
using namespace std;
using namespace boost;

void write_image(string img_name,
    unsigned short width, unsigned short height, 
    unsigned char img[]){
  
  gil::rgb8c_view_t im = interleaved_view(width, height,
      (gil::rgb8_pixel_t*) img,
      width*3*sizeof(unsigned char));

  gil::write_view(img_name, im, gil::png_tag());
}
