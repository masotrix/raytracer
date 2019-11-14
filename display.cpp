#define png_infopp_NULL (png_infopp)NULL 
#define int_p_NULL (int *)NULL
#include <boost/gil/extension/io/png_dynamic_io.hpp>
#include <display.h>
using namespace std;
using namespace boost;

void write_image(string img_name,
    unsigned short width, unsigned short height, 
    unsigned char img[]){
  
  gil::rgb8c_view_t im = interleaved_view(width, height,
      (gil::rgb8_pixel_t*) img,
      width*3*sizeof(unsigned char));

  gil::png_write_view(img_name,im);
}
