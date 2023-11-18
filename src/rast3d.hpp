#ifndef FRAST3D_HPP
#define FRAST3D_HPP
#include <cmath>
#ifdef FRAST3D_IMPLEMENTATION
#include <png++/png.hpp>
#include <iostream>
#include <fstream>
#ifndef FRAST3D_NO_STBI_WRITE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#endif

#define INCLUDE_STB_IMAGE_WRITE_H

#include <stdlib.h>

// if STB_IMAGE_WRITE_STATIC causes problems, try defining STBIWDEF to 'inline' or 'static inline'
#ifndef STBIWDEF
#ifdef STB_IMAGE_WRITE_STATIC
#define STBIWDEF  static
#else
#ifdef __cplusplus
#define STBIWDEF  extern "C" inline
#else
#define STBIWDEF  extern
#endif
#endif
#endif

#ifndef STB_IMAGE_WRITE_STATIC  // C++ forbids static forward declarations
STBIWDEF int stbi_write_tga_with_rle;
STBIWDEF int stbi_write_png_compression_level;
STBIWDEF int stbi_write_force_png_filter;
#endif

#ifndef STBI_WRITE_NO_STDIO
STBIWDEF int stbi_write_png(char const *filename, int w, int h, int comp, const void  *data, int stride_in_bytes);
STBIWDEF int stbi_write_bmp(char const *filename, int w, int h, int comp, const void  *data);
STBIWDEF int stbi_write_tga(char const *filename, int w, int h, int comp, const void  *data);
STBIWDEF int stbi_write_hdr(char const *filename, int w, int h, int comp, const float *data);
STBIWDEF int stbi_write_jpg(char const *filename, int x, int y, int comp, const void  *data, int quality);

#ifdef STBIW_WINDOWS_UTF8
STBIWDEF int stbiw_convert_wchar_to_utf8(char *buffer, size_t bufferlen, const wchar_t* input);
#endif
#endif

typedef void stbi_write_func(void *context, void *data, int size);

STBIWDEF int stbi_write_png_to_func(stbi_write_func *func, void *context, int w, int h, int comp, const void  *data, int stride_in_bytes);
STBIWDEF int stbi_write_bmp_to_func(stbi_write_func *func, void *context, int w, int h, int comp, const void  *data);
STBIWDEF int stbi_write_tga_to_func(stbi_write_func *func, void *context, int w, int h, int comp, const void  *data);
STBIWDEF int stbi_write_hdr_to_func(stbi_write_func *func, void *context, int w, int h, int comp, const float *data);
STBIWDEF int stbi_write_jpg_to_func(stbi_write_func *func, void *context, int x, int y, int comp, const void  *data, int quality);

STBIWDEF void stbi_flip_vertically_on_write(int flip_boolean);

#endif//INCLUDE_STB_IMAGE_WRITE_H

#ifdef STB_IMAGE_WRITE_IMPLEMENTATION

#ifdef _WIN32
   #ifndef _CRT_SECURE_NO_WARNINGS
   #define _CRT_SECURE_NO_WARNINGS
   #endif
   #ifndef _CRT_NONSTDC_NO_DEPRECATE
   #define _CRT_NONSTDC_NO_DEPRECATE
   #endif
#endif

#ifndef STBI_WRITE_NO_STDIO
#include <stdio.h>
#endif // STBI_WRITE_NO_STDIO

#include <stdarg.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#if defined(STBIW_MALLOC) && defined(STBIW_FREE) && (defined(STBIW_REALLOC) || defined(STBIW_REALLOC_SIZED))
// ok
#elif !defined(STBIW_MALLOC) && !defined(STBIW_FREE) && !defined(STBIW_REALLOC) && !defined(STBIW_REALLOC_SIZED)
// ok
#else
#error "Must define all or none of STBIW_MALLOC, STBIW_FREE, and STBIW_REALLOC (or STBIW_REALLOC_SIZED)."
#endif

#ifndef STBIW_MALLOC
#define STBIW_MALLOC(sz)        malloc(sz)
#define STBIW_REALLOC(p,newsz)  realloc(p,newsz)
#define STBIW_FREE(p)           free(p)
#endif

#ifndef STBIW_REALLOC_SIZED
#define STBIW_REALLOC_SIZED(p,oldsz,newsz) STBIW_REALLOC(p,newsz)
#endif


#ifndef STBIW_MEMMOVE
#define STBIW_MEMMOVE(a,b,sz) memmove(a,b,sz)
#endif


#ifndef STBIW_ASSERT
#include <assert.h>
#define STBIW_ASSERT(x) assert(x)
#endif

#define STBIW_UCHAR(x) (unsigned char) ((x) & 0xff)

#ifdef STB_IMAGE_WRITE_STATIC
static int stbi_write_png_compression_level = 8;
static int stbi_write_tga_with_rle = 1;
static int stbi_write_force_png_filter = -1;
#else
int stbi_write_png_compression_level = 8;
int stbi_write_tga_with_rle = 1;
int stbi_write_force_png_filter = -1;
#endif

static int stbi__flip_vertically_on_write = 0;

STBIWDEF void stbi_flip_vertically_on_write(int flag)
{
   stbi__flip_vertically_on_write = flag;
}

typedef struct
{
   stbi_write_func *func;
   void *context;
   unsigned char buffer[64];
   int buf_used;
} stbi__write_context;

// initialize a callback-based context
static void stbi__start_write_callbacks(stbi__write_context *s, stbi_write_func *c, void *context)
{
   s->func    = c;
   s->context = context;
}

#ifndef STBI_WRITE_NO_STDIO

static void stbi__stdio_write(void *context, void *data, int size)
{
   fwrite(data,1,size,(FILE*) context);
}

#if defined(_WIN32) && defined(STBIW_WINDOWS_UTF8)
#ifdef __cplusplus
#define STBIW_EXTERN extern "C"
#else
#define STBIW_EXTERN extern
#endif
STBIW_EXTERN __declspec(dllimport) int __stdcall MultiByteToWideChar(unsigned int cp, unsigned long flags, const char *str, int cbmb, wchar_t *widestr, int cchwide);
STBIW_EXTERN __declspec(dllimport) int __stdcall WideCharToMultiByte(unsigned int cp, unsigned long flags, const wchar_t *widestr, int cchwide, char *str, int cbmb, const char *defchar, int *used_default);

STBIWDEF int stbiw_convert_wchar_to_utf8(char *buffer, size_t bufferlen, const wchar_t* input)
{
   return WideCharToMultiByte(65001 /* UTF8 */, 0, input, -1, buffer, (int) bufferlen, NULL, NULL);
}
#endif

static FILE *stbiw__fopen(char const *filename, char const *mode)
{
   FILE *f;
#if defined(_WIN32) && defined(STBIW_WINDOWS_UTF8)
   wchar_t wMode[64];
   wchar_t wFilename[1024];
   if (0 == MultiByteToWideChar(65001 /* UTF8 */, 0, filename, -1, wFilename, sizeof(wFilename)/sizeof(*wFilename)))
      return 0;

   if (0 == MultiByteToWideChar(65001 /* UTF8 */, 0, mode, -1, wMode, sizeof(wMode)/sizeof(*wMode)))
      return 0;

#if defined(_MSC_VER) && _MSC_VER >= 1400
   if (0 != _wfopen_s(&f, wFilename, wMode))
      f = 0;
#else
   f = _wfopen(wFilename, wMode);
#endif

#elif defined(_MSC_VER) && _MSC_VER >= 1400
   if (0 != fopen_s(&f, filename, mode))
      f=0;
#else
   f = fopen(filename, mode);
#endif
   return f;
}

static int stbi__start_write_file(stbi__write_context *s, const char *filename)
{
   FILE *f = stbiw__fopen(filename, "wb");
   stbi__start_write_callbacks(s, stbi__stdio_write, (void *) f);
   return f != NULL;
}

static void stbi__end_write_file(stbi__write_context *s)
{
   fclose((FILE *)s->context);
}

#endif // !STBI_WRITE_NO_STDIO

typedef unsigned int stbiw_uint32;
typedef int stb_image_write_test[sizeof(stbiw_uint32)==4 ? 1 : -1];

static void stbiw__writefv(stbi__write_context *s, const char *fmt, va_list v)
{
   while (*fmt) {
      switch (*fmt++) {
         case ' ': break;
         case '1': { unsigned char x = STBIW_UCHAR(va_arg(v, int));
                     s->func(s->context,&x,1);
                     break; }
         case '2': { int x = va_arg(v,int);
                     unsigned char b[2];
                     b[0] = STBIW_UCHAR(x);
                     b[1] = STBIW_UCHAR(x>>8);
                     s->func(s->context,b,2);
                     break; }
         case '4': { stbiw_uint32 x = va_arg(v,int);
                     unsigned char b[4];
                     b[0]=STBIW_UCHAR(x);
                     b[1]=STBIW_UCHAR(x>>8);
                     b[2]=STBIW_UCHAR(x>>16);
                     b[3]=STBIW_UCHAR(x>>24);
                     s->func(s->context,b,4);
                     break; }
         default:
            STBIW_ASSERT(0);
            return;
      }
   }
}

static void stbiw__writef(stbi__write_context *s, const char *fmt, ...)
{
   va_list v;
   va_start(v, fmt);
   stbiw__writefv(s, fmt, v);
   va_end(v);
}

static void stbiw__write_flush(stbi__write_context *s)
{
   if (s->buf_used) {
      s->func(s->context, &s->buffer, s->buf_used);
      s->buf_used = 0;
   }
}

static void stbiw__putc(stbi__write_context *s, unsigned char c)
{
   s->func(s->context, &c, 1);
}

static void stbiw__write1(stbi__write_context *s, unsigned char a)
{
   if ((size_t)s->buf_used + 1 > sizeof(s->buffer))
      stbiw__write_flush(s);
   s->buffer[s->buf_used++] = a;
}

static void stbiw__write3(stbi__write_context *s, unsigned char a, unsigned char b, unsigned char c)
{
   int n;
   if ((size_t)s->buf_used + 3 > sizeof(s->buffer))
      stbiw__write_flush(s);
   n = s->buf_used;
   s->buf_used = n+3;
   s->buffer[n+0] = a;
   s->buffer[n+1] = b;
   s->buffer[n+2] = c;
}

static void stbiw__write_pixel(stbi__write_context *s, int rgb_dir, int comp, int write_alpha, int expand_mono, unsigned char *d)
{
   unsigned char bg[3] = { 255, 0, 255}, px[3];
   int k;

   if (write_alpha < 0)
      stbiw__write1(s, d[comp - 1]);

   switch (comp) {
      case 2: // 2 pixels = mono + alpha, alpha is written separately, so same as 1-channel case
      case 1:
         if (expand_mono)
            stbiw__write3(s, d[0], d[0], d[0]); // monochrome bmp
         else
            stbiw__write1(s, d[0]);  // monochrome TGA
         break;
      case 4:
         if (!write_alpha) {
            // composite against pink background
            for (k = 0; k < 3; ++k)
               px[k] = bg[k] + ((d[k] - bg[k]) * d[3]) / 255;
            stbiw__write3(s, px[1 - rgb_dir], px[1], px[1 + rgb_dir]);
            break;
         }
         /* FALLTHROUGH */
      case 3:
         stbiw__write3(s, d[1 - rgb_dir], d[1], d[1 + rgb_dir]);
         break;
   }
   if (write_alpha > 0)
      stbiw__write1(s, d[comp - 1]);
}

static void stbiw__write_pixels(stbi__write_context *s, int rgb_dir, int vdir, int x, int y, int comp, void *data, int write_alpha, int scanline_pad, int expand_mono)
{
   stbiw_uint32 zero = 0;
   int i,j, j_end;

   if (y <= 0)
      return;

   if (stbi__flip_vertically_on_write)
      vdir *= -1;

   if (vdir < 0) {
      j_end = -1; j = y-1;
   } else {
      j_end =  y; j = 0;
   }

   for (; j != j_end; j += vdir) {
      for (i=0; i < x; ++i) {
         unsigned char *d = (unsigned char *) data + (j*x+i)*comp;
         stbiw__write_pixel(s, rgb_dir, comp, write_alpha, expand_mono, d);
      }
      stbiw__write_flush(s);
      s->func(s->context, &zero, scanline_pad);
   }
}

static int stbiw__outfile(stbi__write_context *s, int rgb_dir, int vdir, int x, int y, int comp, int expand_mono, void *data, int alpha, int pad, const char *fmt, ...)
{
   if (y < 0 || x < 0) {
      return 0;
   } else {
      va_list v;
      va_start(v, fmt);
      stbiw__writefv(s, fmt, v);
      va_end(v);
      stbiw__write_pixels(s,rgb_dir,vdir,x,y,comp,data,alpha,pad, expand_mono);
      return 1;
   }
}

static int stbi_write_bmp_core(stbi__write_context *s, int x, int y, int comp, const void *data)
{
   if (comp != 4) {
      // write RGB bitmap
      int pad = (-x*3) & 3;
      return stbiw__outfile(s,-1,-1,x,y,comp,1,(void *) data,0,pad,
              "11 4 22 4" "4 44 22 444444",
              'B', 'M', 14+40+(x*3+pad)*y, 0,0, 14+40,  // file header
               40, x,y, 1,24, 0,0,0,0,0,0);             // bitmap header
   } else {
      // RGBA bitmaps need a v4 header
      // use BI_BITFIELDS mode with 32bpp and alpha mask
      // (straight BI_RGB with alpha mask doesn't work in most readers)
      return stbiw__outfile(s,-1,-1,x,y,comp,1,(void *)data,1,0,
         "11 4 22 4" "4 44 22 444444 4444 4 444 444 444 444",
         'B', 'M', 14+108+x*y*4, 0, 0, 14+108, // file header
         108, x,y, 1,32, 3,0,0,0,0,0, 0xff0000,0xff00,0xff,0xff000000u, 0, 0,0,0, 0,0,0, 0,0,0, 0,0,0); // bitmap V4 header
   }
}

STBIWDEF int stbi_write_bmp_to_func(stbi_write_func *func, void *context, int x, int y, int comp, const void *data)
{
   stbi__write_context s = {0, 0, {0}, 0};
   stbi__start_write_callbacks(&s, func, context);
   return stbi_write_bmp_core(&s, x, y, comp, data);
}

#ifndef STBI_WRITE_NO_STDIO
STBIWDEF int stbi_write_bmp(char const *filename, int x, int y, int comp, const void *data)
{
   stbi__write_context s = {0, 0, {0}, 0};
   if (stbi__start_write_file(&s,filename)) {
      int r = stbi_write_bmp_core(&s, x, y, comp, data);
      stbi__end_write_file(&s);
      return r;
   } else
      return 0;
}
#endif //!STBI_WRITE_NO_STDIO

static int stbi_write_tga_core(stbi__write_context *s, int x, int y, int comp, void *data)
{
   int has_alpha = (comp == 2 || comp == 4);
   int colorbytes = has_alpha ? comp-1 : comp;
   int format = colorbytes < 2 ? 3 : 2; // 3 color channels (RGB/RGBA) = 2, 1 color channel (Y/YA) = 3

   if (y < 0 || x < 0)
      return 0;

   if (!stbi_write_tga_with_rle) {
      return stbiw__outfile(s, -1, -1, x, y, comp, 0, (void *) data, has_alpha, 0,
         "111 221 2222 11", 0, 0, format, 0, 0, 0, 0, 0, x, y, (colorbytes + has_alpha) * 8, has_alpha * 8);
   } else {
      int i,j,k;
      int jend, jdir;

      stbiw__writef(s, "111 221 2222 11", 0,0,format+8, 0,0,0, 0,0,x,y, (colorbytes + has_alpha) * 8, has_alpha * 8);

      if (stbi__flip_vertically_on_write) {
         j = 0;
         jend = y;
         jdir = 1;
      } else {
         j = y-1;
         jend = -1;
         jdir = -1;
      }
      for (; j != jend; j += jdir) {
         unsigned char *row = (unsigned char *) data + j * x * comp;
         int len;

         for (i = 0; i < x; i += len) {
            unsigned char *begin = row + i * comp;
            int diff = 1;
            len = 1;

            if (i < x - 1) {
               ++len;
               diff = memcmp(begin, row + (i + 1) * comp, comp);
               if (diff) {
                  const unsigned char *prev = begin;
                  for (k = i + 2; k < x && len < 128; ++k) {
                     if (memcmp(prev, row + k * comp, comp)) {
                        prev += comp;
                        ++len;
                     } else {
                        --len;
                        break;
                     }
                  }
               } else {
                  for (k = i + 2; k < x && len < 128; ++k) {
                     if (!memcmp(begin, row + k * comp, comp)) {
                        ++len;
                     } else {
                        break;
                     }
                  }
               }
            }

            if (diff) {
               unsigned char header = STBIW_UCHAR(len - 1);
               stbiw__write1(s, header);
               for (k = 0; k < len; ++k) {
                  stbiw__write_pixel(s, -1, comp, has_alpha, 0, begin + k * comp);
               }
            } else {
               unsigned char header = STBIW_UCHAR(len - 129);
               stbiw__write1(s, header);
               stbiw__write_pixel(s, -1, comp, has_alpha, 0, begin);
            }
         }
      }
      stbiw__write_flush(s);
   }
   return 1;
}

STBIWDEF int stbi_write_tga_to_func(stbi_write_func *func, void *context, int x, int y, int comp, const void *data)
{
   stbi__write_context s = {0, 0, {0}, 0};
   stbi__start_write_callbacks(&s, func, context);
   return stbi_write_tga_core(&s, x, y, comp, (void *) data);
}

#ifndef STBI_WRITE_NO_STDIO
STBIWDEF int stbi_write_tga(char const *filename, int x, int y, int comp, const void *data)
{
   stbi__write_context s = {0, 0, {0}, 0};
   if (stbi__start_write_file(&s,filename)) {
      int r = stbi_write_tga_core(&s, x, y, comp, (void *) data);
      stbi__end_write_file(&s);
      return r;
   } else
      return 0;
}
#endif

// *************************************************************************************************
// Radiance RGBE HDR writer
// by Baldur Karlsson

#define stbiw__max(a, b)  ((a) > (b) ? (a) : (b))

#ifndef STBI_WRITE_NO_STDIO

static void stbiw__linear_to_rgbe(unsigned char *rgbe, float *linear)
{
   int exponent;
   float maxcomp = stbiw__max(linear[0], stbiw__max(linear[1], linear[2]));

   if (maxcomp < 1e-32f) {
      rgbe[0] = rgbe[1] = rgbe[2] = rgbe[3] = 0;
   } else {
      float normalize = (float) frexp(maxcomp, &exponent) * 256.0f/maxcomp;

      rgbe[0] = (unsigned char)(linear[0] * normalize);
      rgbe[1] = (unsigned char)(linear[1] * normalize);
      rgbe[2] = (unsigned char)(linear[2] * normalize);
      rgbe[3] = (unsigned char)(exponent + 128);
   }
}

static void stbiw__write_run_data(stbi__write_context *s, int length, unsigned char databyte)
{
   unsigned char lengthbyte = STBIW_UCHAR(length+128);
   STBIW_ASSERT(length+128 <= 255);
   s->func(s->context, &lengthbyte, 1);
   s->func(s->context, &databyte, 1);
}

static void stbiw__write_dump_data(stbi__write_context *s, int length, unsigned char *data)
{
   unsigned char lengthbyte = STBIW_UCHAR(length);
   STBIW_ASSERT(length <= 128); // inconsistent with spec but consistent with official code
   s->func(s->context, &lengthbyte, 1);
   s->func(s->context, data, length);
}

static void stbiw__write_hdr_scanline(stbi__write_context *s, int width, int ncomp, unsigned char *scratch, float *scanline)
{
   unsigned char scanlineheader[4] = { 2, 2, 0, 0 };
   unsigned char rgbe[4];
   float linear[3];
   int x;

   scanlineheader[2] = (width&0xff00)>>8;
   scanlineheader[3] = (width&0x00ff);

   /* skip RLE for images too small or large */
   if (width < 8 || width >= 32768) {
      for (x=0; x < width; x++) {
         switch (ncomp) {
            case 4: /* fallthrough */
            case 3: linear[2] = scanline[x*ncomp + 2];
                    linear[1] = scanline[x*ncomp + 1];
                    linear[0] = scanline[x*ncomp + 0];
                    break;
            default:
                    linear[0] = linear[1] = linear[2] = scanline[x*ncomp + 0];
                    break;
         }
         stbiw__linear_to_rgbe(rgbe, linear);
         s->func(s->context, rgbe, 4);
      }
   } else {
      int c,r;
      /* encode into scratch buffer */
      for (x=0; x < width; x++) {
         switch(ncomp) {
            case 4: /* fallthrough */
            case 3: linear[2] = scanline[x*ncomp + 2];
                    linear[1] = scanline[x*ncomp + 1];
                    linear[0] = scanline[x*ncomp + 0];
                    break;
            default:
                    linear[0] = linear[1] = linear[2] = scanline[x*ncomp + 0];
                    break;
         }
         stbiw__linear_to_rgbe(rgbe, linear);
         scratch[x + width*0] = rgbe[0];
         scratch[x + width*1] = rgbe[1];
         scratch[x + width*2] = rgbe[2];
         scratch[x + width*3] = rgbe[3];
      }

      s->func(s->context, scanlineheader, 4);

      /* RLE each component separately */
      for (c=0; c < 4; c++) {
         unsigned char *comp = &scratch[width*c];

         x = 0;
         while (x < width) {
            // find first run
            r = x;
            while (r+2 < width) {
               if (comp[r] == comp[r+1] && comp[r] == comp[r+2])
                  break;
               ++r;
            }
            if (r+2 >= width)
               r = width;
            // dump up to first run
            while (x < r) {
               int len = r-x;
               if (len > 128) len = 128;
               stbiw__write_dump_data(s, len, &comp[x]);
               x += len;
            }
            // if there's a run, output it
            if (r+2 < width) { // same test as what we break out of in search loop, so only true if we break'd
               // find next byte after run
               while (r < width && comp[r] == comp[x])
                  ++r;
               // output run up to r
               while (x < r) {
                  int len = r-x;
                  if (len > 127) len = 127;
                  stbiw__write_run_data(s, len, comp[x]);
                  x += len;
               }
            }
         }
      }
   }
}

static int stbi_write_hdr_core(stbi__write_context *s, int x, int y, int comp, float *data)
{
   if (y <= 0 || x <= 0 || data == NULL)
      return 0;
   else {
      // Each component is stored separately. Allocate scratch space for full output scanline.
      unsigned char *scratch = (unsigned char *) STBIW_MALLOC(x*4);
      int i, len;
      char buffer[128];
      char header[] = "#?RADIANCE\n# Written by stb_image_write.h\nFORMAT=32-bit_rle_rgbe\n";
      s->func(s->context, header, sizeof(header)-1);

#ifdef __STDC_LIB_EXT1__
      len = sprintf_s(buffer, sizeof(buffer), "EXPOSURE=          1.0000000000000\n\n-Y %d +X %d\n", y, x);
#else
      len = sprintf(buffer, "EXPOSURE=          1.0000000000000\n\n-Y %d +X %d\n", y, x);
#endif
      s->func(s->context, buffer, len);

      for(i=0; i < y; i++)
         stbiw__write_hdr_scanline(s, x, comp, scratch, data + comp*x*(stbi__flip_vertically_on_write ? y-1-i : i));
      STBIW_FREE(scratch);
      return 1;
   }
}

STBIWDEF int stbi_write_hdr_to_func(stbi_write_func *func, void *context, int x, int y, int comp, const float *data)
{
   stbi__write_context s = {0, 0, {0}, 0};
   stbi__start_write_callbacks(&s, func, context);
   return stbi_write_hdr_core(&s, x, y, comp, (float *) data);
}

STBIWDEF int stbi_write_hdr(char const *filename, int x, int y, int comp, const float *data)
{
   stbi__write_context s = {0, 0, {0}, 0};
   if (stbi__start_write_file(&s,filename)) {
      int r = stbi_write_hdr_core(&s, x, y, comp, (float *) data);
      stbi__end_write_file(&s);
      return r;
   } else
      return 0;
}
#endif // STBI_WRITE_NO_STDIO


//////////////////////////////////////////////////////////////////////////////
//
// PNG writer
//

#ifndef STBIW_ZLIB_COMPRESS
// stretchy buffer; stbiw__sbpush() == vector<>::push_back() -- stbiw__sbcount() == vector<>::size()
#define stbiw__sbraw(a) ((int *) (void *) (a) - 2)
#define stbiw__sbm(a)   stbiw__sbraw(a)[0]
#define stbiw__sbn(a)   stbiw__sbraw(a)[1]

#define stbiw__sbneedgrow(a,n)  ((a)==0 || stbiw__sbn(a)+n >= stbiw__sbm(a))
#define stbiw__sbmaybegrow(a,n) (stbiw__sbneedgrow(a,(n)) ? stbiw__sbgrow(a,n) : 0)
#define stbiw__sbgrow(a,n)  stbiw__sbgrowf((void **) &(a), (n), sizeof(*(a)))

#define stbiw__sbpush(a, v)      (stbiw__sbmaybegrow(a,1), (a)[stbiw__sbn(a)++] = (v))
#define stbiw__sbcount(a)        ((a) ? stbiw__sbn(a) : 0)
#define stbiw__sbfree(a)         ((a) ? STBIW_FREE(stbiw__sbraw(a)),0 : 0)

static void *stbiw__sbgrowf(void **arr, int increment, int itemsize)
{
   int m = *arr ? 2*stbiw__sbm(*arr)+increment : increment+1;
   void *p = STBIW_REALLOC_SIZED(*arr ? stbiw__sbraw(*arr) : 0, *arr ? (stbiw__sbm(*arr)*itemsize + sizeof(int)*2) : 0, itemsize * m + sizeof(int)*2);
   STBIW_ASSERT(p);
   if (p) {
      if (!*arr) ((int *) p)[1] = 0;
      *arr = (void *) ((int *) p + 2);
      stbiw__sbm(*arr) = m;
   }
   return *arr;
}

static unsigned char *stbiw__zlib_flushf(unsigned char *data, unsigned int *bitbuffer, int *bitcount)
{
   while (*bitcount >= 8) {
      stbiw__sbpush(data, STBIW_UCHAR(*bitbuffer));
      *bitbuffer >>= 8;
      *bitcount -= 8;
   }
   return data;
}

static int stbiw__zlib_bitrev(int code, int codebits)
{
   int res=0;
   while (codebits--) {
      res = (res << 1) | (code & 1);
      code >>= 1;
   }
   return res;
}

static unsigned int stbiw__zlib_countm(unsigned char *a, unsigned char *b, int limit)
{
   int i;
   for (i=0; i < limit && i < 258; ++i)
      if (a[i] != b[i]) break;
   return i;
}

static unsigned int stbiw__zhash(unsigned char *data)
{
   stbiw_uint32 hash = data[0] + (data[1] << 8) + (data[2] << 16);
   hash ^= hash << 3;
   hash += hash >> 5;
   hash ^= hash << 4;
   hash += hash >> 17;
   hash ^= hash << 25;
   hash += hash >> 6;
   return hash;
}

#define stbiw__zlib_flush() (out = stbiw__zlib_flushf(out, &bitbuf, &bitcount))
#define stbiw__zlib_add(code,codebits) \
      (bitbuf |= (code) << bitcount, bitcount += (codebits), stbiw__zlib_flush())
#define stbiw__zlib_huffa(b,c)  stbiw__zlib_add(stbiw__zlib_bitrev(b,c),c)
// default huffman tables
#define stbiw__zlib_huff1(n)  stbiw__zlib_huffa(0x30 + (n), 8)
#define stbiw__zlib_huff2(n)  stbiw__zlib_huffa(0x190 + (n)-144, 9)
#define stbiw__zlib_huff3(n)  stbiw__zlib_huffa(0 + (n)-256,7)
#define stbiw__zlib_huff4(n)  stbiw__zlib_huffa(0xc0 + (n)-280,8)
#define stbiw__zlib_huff(n)  ((n) <= 143 ? stbiw__zlib_huff1(n) : (n) <= 255 ? stbiw__zlib_huff2(n) : (n) <= 279 ? stbiw__zlib_huff3(n) : stbiw__zlib_huff4(n))
#define stbiw__zlib_huffb(n) ((n) <= 143 ? stbiw__zlib_huff1(n) : stbiw__zlib_huff2(n))

#define stbiw__ZHASH   16384

#endif // STBIW_ZLIB_COMPRESS

STBIWDEF unsigned char * stbi_zlib_compress(unsigned char *data, int data_len, int *out_len, int quality)
{
#ifdef STBIW_ZLIB_COMPRESS
   // user provided a zlib compress implementation, use that
   return STBIW_ZLIB_COMPRESS(data, data_len, out_len, quality);
#else // use builtin
   static unsigned short lengthc[] = { 3,4,5,6,7,8,9,10,11,13,15,17,19,23,27,31,35,43,51,59,67,83,99,115,131,163,195,227,258, 259 };
   static unsigned char  lengtheb[]= { 0,0,0,0,0,0,0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4,  4,  5,  5,  5,  5,  0 };
   static unsigned short distc[]   = { 1,2,3,4,5,7,9,13,17,25,33,49,65,97,129,193,257,385,513,769,1025,1537,2049,3073,4097,6145,8193,12289,16385,24577, 32768 };
   static unsigned char  disteb[]  = { 0,0,0,0,1,1,2,2,3,3,4,4,5,5,6,6,7,7,8,8,9,9,10,10,11,11,12,12,13,13 };
   unsigned int bitbuf=0;
   int i,j, bitcount=0;
   unsigned char *out = NULL;
   unsigned char ***hash_table = (unsigned char***) STBIW_MALLOC(stbiw__ZHASH * sizeof(unsigned char**));
   if (hash_table == NULL)
      return NULL;
   if (quality < 5) quality = 5;

   stbiw__sbpush(out, 0x78);   // DEFLATE 32K window
   stbiw__sbpush(out, 0x5e);   // FLEVEL = 1
   stbiw__zlib_add(1,1);  // BFINAL = 1
   stbiw__zlib_add(1,2);  // BTYPE = 1 -- fixed huffman

   for (i=0; i < stbiw__ZHASH; ++i)
      hash_table[i] = NULL;

   i=0;
   while (i < data_len-3) {
      // hash next 3 bytes of data to be compressed
      int h = stbiw__zhash(data+i)&(stbiw__ZHASH-1), best=3;
      unsigned char *bestloc = 0;
      unsigned char **hlist = hash_table[h];
      int n = stbiw__sbcount(hlist);
      for (j=0; j < n; ++j) {
         if (hlist[j]-data > i-32768) { // if entry lies within window
            int d = stbiw__zlib_countm(hlist[j], data+i, data_len-i);
            if (d >= best) { best=d; bestloc=hlist[j]; }
         }
      }
      // when hash table entry is too long, delete half the entries
      if (hash_table[h] && stbiw__sbn(hash_table[h]) == 2*quality) {
         STBIW_MEMMOVE(hash_table[h], hash_table[h]+quality, sizeof(hash_table[h][0])*quality);
         stbiw__sbn(hash_table[h]) = quality;
      }
      stbiw__sbpush(hash_table[h],data+i);

      if (bestloc) {
         // "lazy matching" - check match at *next* byte, and if it's better, do cur byte as literal
         h = stbiw__zhash(data+i+1)&(stbiw__ZHASH-1);
         hlist = hash_table[h];
         n = stbiw__sbcount(hlist);
         for (j=0; j < n; ++j) {
            if (hlist[j]-data > i-32767) {
               int e = stbiw__zlib_countm(hlist[j], data+i+1, data_len-i-1);
               if (e > best) { // if next match is better, bail on current match
                  bestloc = NULL;
                  break;
               }
            }
         }
      }

      if (bestloc) {
         int d = (int) (data+i - bestloc); // distance back
         STBIW_ASSERT(d <= 32767 && best <= 258);
         for (j=0; best > lengthc[j+1]-1; ++j);
         stbiw__zlib_huff(j+257);
         if (lengtheb[j]) stbiw__zlib_add(best - lengthc[j], lengtheb[j]);
         for (j=0; d > distc[j+1]-1; ++j);
         stbiw__zlib_add(stbiw__zlib_bitrev(j,5),5);
         if (disteb[j]) stbiw__zlib_add(d - distc[j], disteb[j]);
         i += best;
      } else {
         stbiw__zlib_huffb(data[i]);
         ++i;
      }
   }
   // write out final bytes
   for (;i < data_len; ++i)
      stbiw__zlib_huffb(data[i]);
   stbiw__zlib_huff(256); // end of block
   // pad with 0 bits to byte boundary
   while (bitcount)
      stbiw__zlib_add(0,1);

   for (i=0; i < stbiw__ZHASH; ++i)
      (void) stbiw__sbfree(hash_table[i]);
   STBIW_FREE(hash_table);

   // store uncompressed instead if compression was worse
   if (stbiw__sbn(out) > data_len + 2 + ((data_len+32766)/32767)*5) {
      stbiw__sbn(out) = 2;  // truncate to DEFLATE 32K window and FLEVEL = 1
      for (j = 0; j < data_len;) {
         int blocklen = data_len - j;
         if (blocklen > 32767) blocklen = 32767;
         stbiw__sbpush(out, data_len - j == blocklen); // BFINAL = ?, BTYPE = 0 -- no compression
         stbiw__sbpush(out, STBIW_UCHAR(blocklen)); // LEN
         stbiw__sbpush(out, STBIW_UCHAR(blocklen >> 8));
         stbiw__sbpush(out, STBIW_UCHAR(~blocklen)); // NLEN
         stbiw__sbpush(out, STBIW_UCHAR(~blocklen >> 8));
         memcpy(out+stbiw__sbn(out), data+j, blocklen);
         stbiw__sbn(out) += blocklen;
         j += blocklen;
      }
   }

   {
      // compute adler32 on input
      unsigned int s1=1, s2=0;
      int blocklen = (int) (data_len % 5552);
      j=0;
      while (j < data_len) {
         for (i=0; i < blocklen; ++i) { s1 += data[j+i]; s2 += s1; }
         s1 %= 65521; s2 %= 65521;
         j += blocklen;
         blocklen = 5552;
      }
      stbiw__sbpush(out, STBIW_UCHAR(s2 >> 8));
      stbiw__sbpush(out, STBIW_UCHAR(s2));
      stbiw__sbpush(out, STBIW_UCHAR(s1 >> 8));
      stbiw__sbpush(out, STBIW_UCHAR(s1));
   }
   *out_len = stbiw__sbn(out);
   // make returned pointer freeable
   STBIW_MEMMOVE(stbiw__sbraw(out), out, *out_len);
   return (unsigned char *) stbiw__sbraw(out);
#endif // STBIW_ZLIB_COMPRESS
}

static unsigned int stbiw__crc32(unsigned char *buffer, int len)
{
#ifdef STBIW_CRC32
    return STBIW_CRC32(buffer, len);
#else
   static unsigned int crc_table[256] =
   {
      0x00000000, 0x77073096, 0xEE0E612C, 0x990951BA, 0x076DC419, 0x706AF48F, 0xE963A535, 0x9E6495A3,
      0x0eDB8832, 0x79DCB8A4, 0xE0D5E91E, 0x97D2D988, 0x09B64C2B, 0x7EB17CBD, 0xE7B82D07, 0x90BF1D91,
      0x1DB71064, 0x6AB020F2, 0xF3B97148, 0x84BE41DE, 0x1ADAD47D, 0x6DDDE4EB, 0xF4D4B551, 0x83D385C7,
      0x136C9856, 0x646BA8C0, 0xFD62F97A, 0x8A65C9EC, 0x14015C4F, 0x63066CD9, 0xFA0F3D63, 0x8D080DF5,
      0x3B6E20C8, 0x4C69105E, 0xD56041E4, 0xA2677172, 0x3C03E4D1, 0x4B04D447, 0xD20D85FD, 0xA50AB56B,
      0x35B5A8FA, 0x42B2986C, 0xDBBBC9D6, 0xACBCF940, 0x32D86CE3, 0x45DF5C75, 0xDCD60DCF, 0xABD13D59,
      0x26D930AC, 0x51DE003A, 0xC8D75180, 0xBFD06116, 0x21B4F4B5, 0x56B3C423, 0xCFBA9599, 0xB8BDA50F,
      0x2802B89E, 0x5F058808, 0xC60CD9B2, 0xB10BE924, 0x2F6F7C87, 0x58684C11, 0xC1611DAB, 0xB6662D3D,
      0x76DC4190, 0x01DB7106, 0x98D220BC, 0xEFD5102A, 0x71B18589, 0x06B6B51F, 0x9FBFE4A5, 0xE8B8D433,
      0x7807C9A2, 0x0F00F934, 0x9609A88E, 0xE10E9818, 0x7F6A0DBB, 0x086D3D2D, 0x91646C97, 0xE6635C01,
      0x6B6B51F4, 0x1C6C6162, 0x856530D8, 0xF262004E, 0x6C0695ED, 0x1B01A57B, 0x8208F4C1, 0xF50FC457,
      0x65B0D9C6, 0x12B7E950, 0x8BBEB8EA, 0xFCB9887C, 0x62DD1DDF, 0x15DA2D49, 0x8CD37CF3, 0xFBD44C65,
      0x4DB26158, 0x3AB551CE, 0xA3BC0074, 0xD4BB30E2, 0x4ADFA541, 0x3DD895D7, 0xA4D1C46D, 0xD3D6F4FB,
      0x4369E96A, 0x346ED9FC, 0xAD678846, 0xDA60B8D0, 0x44042D73, 0x33031DE5, 0xAA0A4C5F, 0xDD0D7CC9,
      0x5005713C, 0x270241AA, 0xBE0B1010, 0xC90C2086, 0x5768B525, 0x206F85B3, 0xB966D409, 0xCE61E49F,
      0x5EDEF90E, 0x29D9C998, 0xB0D09822, 0xC7D7A8B4, 0x59B33D17, 0x2EB40D81, 0xB7BD5C3B, 0xC0BA6CAD,
      0xEDB88320, 0x9ABFB3B6, 0x03B6E20C, 0x74B1D29A, 0xEAD54739, 0x9DD277AF, 0x04DB2615, 0x73DC1683,
      0xE3630B12, 0x94643B84, 0x0D6D6A3E, 0x7A6A5AA8, 0xE40ECF0B, 0x9309FF9D, 0x0A00AE27, 0x7D079EB1,
      0xF00F9344, 0x8708A3D2, 0x1E01F268, 0x6906C2FE, 0xF762575D, 0x806567CB, 0x196C3671, 0x6E6B06E7,
      0xFED41B76, 0x89D32BE0, 0x10DA7A5A, 0x67DD4ACC, 0xF9B9DF6F, 0x8EBEEFF9, 0x17B7BE43, 0x60B08ED5,
      0xD6D6A3E8, 0xA1D1937E, 0x38D8C2C4, 0x4FDFF252, 0xD1BB67F1, 0xA6BC5767, 0x3FB506DD, 0x48B2364B,
      0xD80D2BDA, 0xAF0A1B4C, 0x36034AF6, 0x41047A60, 0xDF60EFC3, 0xA867DF55, 0x316E8EEF, 0x4669BE79,
      0xCB61B38C, 0xBC66831A, 0x256FD2A0, 0x5268E236, 0xCC0C7795, 0xBB0B4703, 0x220216B9, 0x5505262F,
      0xC5BA3BBE, 0xB2BD0B28, 0x2BB45A92, 0x5CB36A04, 0xC2D7FFA7, 0xB5D0CF31, 0x2CD99E8B, 0x5BDEAE1D,
      0x9B64C2B0, 0xEC63F226, 0x756AA39C, 0x026D930A, 0x9C0906A9, 0xEB0E363F, 0x72076785, 0x05005713,
      0x95BF4A82, 0xE2B87A14, 0x7BB12BAE, 0x0CB61B38, 0x92D28E9B, 0xE5D5BE0D, 0x7CDCEFB7, 0x0BDBDF21,
      0x86D3D2D4, 0xF1D4E242, 0x68DDB3F8, 0x1FDA836E, 0x81BE16CD, 0xF6B9265B, 0x6FB077E1, 0x18B74777,
      0x88085AE6, 0xFF0F6A70, 0x66063BCA, 0x11010B5C, 0x8F659EFF, 0xF862AE69, 0x616BFFD3, 0x166CCF45,
      0xA00AE278, 0xD70DD2EE, 0x4E048354, 0x3903B3C2, 0xA7672661, 0xD06016F7, 0x4969474D, 0x3E6E77DB,
      0xAED16A4A, 0xD9D65ADC, 0x40DF0B66, 0x37D83BF0, 0xA9BCAE53, 0xDEBB9EC5, 0x47B2CF7F, 0x30B5FFE9,
      0xBDBDF21C, 0xCABAC28A, 0x53B39330, 0x24B4A3A6, 0xBAD03605, 0xCDD70693, 0x54DE5729, 0x23D967BF,
      0xB3667A2E, 0xC4614AB8, 0x5D681B02, 0x2A6F2B94, 0xB40BBE37, 0xC30C8EA1, 0x5A05DF1B, 0x2D02EF8D
   };

   unsigned int crc = ~0u;
   int i;
   for (i=0; i < len; ++i)
      crc = (crc >> 8) ^ crc_table[buffer[i] ^ (crc & 0xff)];
   return ~crc;
#endif
}

#define stbiw__wpng4(o,a,b,c,d) ((o)[0]=STBIW_UCHAR(a),(o)[1]=STBIW_UCHAR(b),(o)[2]=STBIW_UCHAR(c),(o)[3]=STBIW_UCHAR(d),(o)+=4)
#define stbiw__wp32(data,v) stbiw__wpng4(data, (v)>>24,(v)>>16,(v)>>8,(v));
#define stbiw__wptag(data,s) stbiw__wpng4(data, s[0],s[1],s[2],s[3])

static void stbiw__wpcrc(unsigned char **data, int len)
{
   unsigned int crc = stbiw__crc32(*data - len - 4, len+4);
   stbiw__wp32(*data, crc);
}

static unsigned char stbiw__paeth(int a, int b, int c)
{
   int p = a + b - c, pa = abs(p-a), pb = abs(p-b), pc = abs(p-c);
   if (pa <= pb && pa <= pc) return STBIW_UCHAR(a);
   if (pb <= pc) return STBIW_UCHAR(b);
   return STBIW_UCHAR(c);
}

// @OPTIMIZE: provide an option that always forces left-predict or paeth predict
static void stbiw__encode_png_line(unsigned char *pixels, int stride_bytes, int width, int height, int y, int n, int filter_type, signed char *line_buffer)
{
   static int mapping[] = { 0,1,2,3,4 };
   static int firstmap[] = { 0,1,0,5,6 };
   int *mymap = (y != 0) ? mapping : firstmap;
   int i;
   int type = mymap[filter_type];
   unsigned char *z = pixels + stride_bytes * (stbi__flip_vertically_on_write ? height-1-y : y);
   int signed_stride = stbi__flip_vertically_on_write ? -stride_bytes : stride_bytes;

   if (type==0) {
      memcpy(line_buffer, z, width*n);
      return;
   }

   // first loop isn't optimized since it's just one pixel
   for (i = 0; i < n; ++i) {
      switch (type) {
         case 1: line_buffer[i] = z[i]; break;
         case 2: line_buffer[i] = z[i] - z[i-signed_stride]; break;
         case 3: line_buffer[i] = z[i] - (z[i-signed_stride]>>1); break;
         case 4: line_buffer[i] = (signed char) (z[i] - stbiw__paeth(0,z[i-signed_stride],0)); break;
         case 5: line_buffer[i] = z[i]; break;
         case 6: line_buffer[i] = z[i]; break;
      }
   }
   switch (type) {
      case 1: for (i=n; i < width*n; ++i) line_buffer[i] = z[i] - z[i-n]; break;
      case 2: for (i=n; i < width*n; ++i) line_buffer[i] = z[i] - z[i-signed_stride]; break;
      case 3: for (i=n; i < width*n; ++i) line_buffer[i] = z[i] - ((z[i-n] + z[i-signed_stride])>>1); break;
      case 4: for (i=n; i < width*n; ++i) line_buffer[i] = z[i] - stbiw__paeth(z[i-n], z[i-signed_stride], z[i-signed_stride-n]); break;
      case 5: for (i=n; i < width*n; ++i) line_buffer[i] = z[i] - (z[i-n]>>1); break;
      case 6: for (i=n; i < width*n; ++i) line_buffer[i] = z[i] - stbiw__paeth(z[i-n], 0,0); break;
   }
}

STBIWDEF unsigned char *stbi_write_png_to_mem(const unsigned char *pixels, int stride_bytes, int x, int y, int n, int *out_len)
{
   int force_filter = stbi_write_force_png_filter;
   int ctype[5] = { -1, 0, 4, 2, 6 };
   unsigned char sig[8] = { 137,80,78,71,13,10,26,10 };
   unsigned char *out,*o, *filt, *zlib;
   signed char *line_buffer;
   int j,zlen;

   if (stride_bytes == 0)
      stride_bytes = x * n;

   if (force_filter >= 5) {
      force_filter = -1;
   }

   filt = (unsigned char *) STBIW_MALLOC((x*n+1) * y); if (!filt) return 0;
   line_buffer = (signed char *) STBIW_MALLOC(x * n); if (!line_buffer) { STBIW_FREE(filt); return 0; }
   for (j=0; j < y; ++j) {
      int filter_type;
      if (force_filter > -1) {
         filter_type = force_filter;
         stbiw__encode_png_line((unsigned char*)(pixels), stride_bytes, x, y, j, n, force_filter, line_buffer);
      } else { // Estimate the best filter by running through all of them:
         int best_filter = 0, best_filter_val = 0x7fffffff, est, i;
         for (filter_type = 0; filter_type < 5; filter_type++) {
            stbiw__encode_png_line((unsigned char*)(pixels), stride_bytes, x, y, j, n, filter_type, line_buffer);

            // Estimate the entropy of the line using this filter; the less, the better.
            est = 0;
            for (i = 0; i < x*n; ++i) {
               est += abs((signed char) line_buffer[i]);
            }
            if (est < best_filter_val) {
               best_filter_val = est;
               best_filter = filter_type;
            }
         }
         if (filter_type != best_filter) {  // If the last iteration already got us the best filter, don't redo it
            stbiw__encode_png_line((unsigned char*)(pixels), stride_bytes, x, y, j, n, best_filter, line_buffer);
            filter_type = best_filter;
         }
      }
      // when we get here, filter_type contains the filter type, and line_buffer contains the data
      filt[j*(x*n+1)] = (unsigned char) filter_type;
      STBIW_MEMMOVE(filt+j*(x*n+1)+1, line_buffer, x*n);
   }
   STBIW_FREE(line_buffer);
   zlib = stbi_zlib_compress(filt, y*( x*n+1), &zlen, stbi_write_png_compression_level);
   STBIW_FREE(filt);
   if (!zlib) return 0;

   // each tag requires 12 bytes of overhead
   out = (unsigned char *) STBIW_MALLOC(8 + 12+13 + 12+zlen + 12);
   if (!out) return 0;
   *out_len = 8 + 12+13 + 12+zlen + 12;

   o=out;
   STBIW_MEMMOVE(o,sig,8); o+= 8;
   stbiw__wp32(o, 13); // header length
   stbiw__wptag(o, "IHDR");
   stbiw__wp32(o, x);
   stbiw__wp32(o, y);
   *o++ = 8;
   *o++ = STBIW_UCHAR(ctype[n]);
   *o++ = 0;
   *o++ = 0;
   *o++ = 0;
   stbiw__wpcrc(&o,13);

   stbiw__wp32(o, zlen);
   stbiw__wptag(o, "IDAT");
   STBIW_MEMMOVE(o, zlib, zlen);
   o += zlen;
   STBIW_FREE(zlib);
   stbiw__wpcrc(&o, zlen);

   stbiw__wp32(o,0);
   stbiw__wptag(o, "IEND");
   stbiw__wpcrc(&o,0);

   STBIW_ASSERT(o == out + *out_len);

   return out;
}

#ifndef STBI_WRITE_NO_STDIO
STBIWDEF int stbi_write_png(char const *filename, int x, int y, int comp, const void *data, int stride_bytes)
{
   FILE *f;
   int len;
   unsigned char *png = stbi_write_png_to_mem((const unsigned char *) data, stride_bytes, x, y, comp, &len);
   if (png == NULL) return 0;

   f = stbiw__fopen(filename, "wb");
   if (!f) { STBIW_FREE(png); return 0; }
   fwrite(png, 1, len, f);
   fclose(f);
   STBIW_FREE(png);
   return 1;
}
#endif

STBIWDEF int stbi_write_png_to_func(stbi_write_func *func, void *context, int x, int y, int comp, const void *data, int stride_bytes)
{
   int len;
   unsigned char *png = stbi_write_png_to_mem((const unsigned char *) data, stride_bytes, x, y, comp, &len);
   if (png == NULL) return 0;
   func(context, png, len);
   STBIW_FREE(png);
   return 1;
}


/* ***************************************************************************
 *
 * JPEG writer
 *
 * This is based on Jon Olick's jo_jpeg.cpp:
 * public domain Simple, Minimalistic JPEG writer - http://www.jonolick.com/code.html
 */

static const unsigned char stbiw__jpg_ZigZag[] = { 0,1,5,6,14,15,27,28,2,4,7,13,16,26,29,42,3,8,12,17,25,30,41,43,9,11,18,
      24,31,40,44,53,10,19,23,32,39,45,52,54,20,22,33,38,46,51,55,60,21,34,37,47,50,56,59,61,35,36,48,49,57,58,62,63 };

static void stbiw__jpg_writeBits(stbi__write_context *s, int *bitBufP, int *bitCntP, const unsigned short *bs) {
   int bitBuf = *bitBufP, bitCnt = *bitCntP;
   bitCnt += bs[1];
   bitBuf |= bs[0] << (24 - bitCnt);
   while(bitCnt >= 8) {
      unsigned char c = (bitBuf >> 16) & 255;
      stbiw__putc(s, c);
      if(c == 255) {
         stbiw__putc(s, 0);
      }
      bitBuf <<= 8;
      bitCnt -= 8;
   }
   *bitBufP = bitBuf;
   *bitCntP = bitCnt;
}

static void stbiw__jpg_DCT(float *d0p, float *d1p, float *d2p, float *d3p, float *d4p, float *d5p, float *d6p, float *d7p) {
   float d0 = *d0p, d1 = *d1p, d2 = *d2p, d3 = *d3p, d4 = *d4p, d5 = *d5p, d6 = *d6p, d7 = *d7p;
   float z1, z2, z3, z4, z5, z11, z13;

   float tmp0 = d0 + d7;
   float tmp7 = d0 - d7;
   float tmp1 = d1 + d6;
   float tmp6 = d1 - d6;
   float tmp2 = d2 + d5;
   float tmp5 = d2 - d5;
   float tmp3 = d3 + d4;
   float tmp4 = d3 - d4;

   // Even part
   float tmp10 = tmp0 + tmp3;   // phase 2
   float tmp13 = tmp0 - tmp3;
   float tmp11 = tmp1 + tmp2;
   float tmp12 = tmp1 - tmp2;

   d0 = tmp10 + tmp11;       // phase 3
   d4 = tmp10 - tmp11;

   z1 = (tmp12 + tmp13) * 0.707106781f; // c4
   d2 = tmp13 + z1;       // phase 5
   d6 = tmp13 - z1;

   // Odd part
   tmp10 = tmp4 + tmp5;       // phase 2
   tmp11 = tmp5 + tmp6;
   tmp12 = tmp6 + tmp7;

   // The rotator is modified from fig 4-8 to avoid extra negations.
   z5 = (tmp10 - tmp12) * 0.382683433f; // c6
   z2 = tmp10 * 0.541196100f + z5; // c2-c6
   z4 = tmp12 * 1.306562965f + z5; // c2+c6
   z3 = tmp11 * 0.707106781f; // c4

   z11 = tmp7 + z3;      // phase 5
   z13 = tmp7 - z3;

   *d5p = z13 + z2;         // phase 6
   *d3p = z13 - z2;
   *d1p = z11 + z4;
   *d7p = z11 - z4;

   *d0p = d0;  *d2p = d2;  *d4p = d4;  *d6p = d6;
}

static void stbiw__jpg_calcBits(int val, unsigned short bits[2]) {
   int tmp1 = val < 0 ? -val : val;
   val = val < 0 ? val-1 : val;
   bits[1] = 1;
   while(tmp1 >>= 1) {
      ++bits[1];
   }
   bits[0] = val & ((1<<bits[1])-1);
}

static int stbiw__jpg_processDU(stbi__write_context *s, int *bitBuf, int *bitCnt, float *CDU, int du_stride, float *fdtbl, int DC, const unsigned short HTDC[256][2], const unsigned short HTAC[256][2]) {
   const unsigned short EOB[2] = { HTAC[0x00][0], HTAC[0x00][1] };
   const unsigned short M16zeroes[2] = { HTAC[0xF0][0], HTAC[0xF0][1] };
   int dataOff, i, j, n, diff, end0pos, x, y;
   int DU[64];

   // DCT rows
   for(dataOff=0, n=du_stride*8; dataOff<n; dataOff+=du_stride) {
      stbiw__jpg_DCT(&CDU[dataOff], &CDU[dataOff+1], &CDU[dataOff+2], &CDU[dataOff+3], &CDU[dataOff+4], &CDU[dataOff+5], &CDU[dataOff+6], &CDU[dataOff+7]);
   }
   // DCT columns
   for(dataOff=0; dataOff<8; ++dataOff) {
      stbiw__jpg_DCT(&CDU[dataOff], &CDU[dataOff+du_stride], &CDU[dataOff+du_stride*2], &CDU[dataOff+du_stride*3], &CDU[dataOff+du_stride*4],
                     &CDU[dataOff+du_stride*5], &CDU[dataOff+du_stride*6], &CDU[dataOff+du_stride*7]);
   }
   // Quantize/descale/zigzag the coefficients
   for(y = 0, j=0; y < 8; ++y) {
      for(x = 0; x < 8; ++x,++j) {
         float v;
         i = y*du_stride+x;
         v = CDU[i]*fdtbl[j];
         // DU[stbiw__jpg_ZigZag[j]] = (int)(v < 0 ? ceilf(v - 0.5f) : floorf(v + 0.5f));
         // ceilf() and floorf() are C99, not C89, but I /think/ they're not needed here anyway?
         DU[stbiw__jpg_ZigZag[j]] = (int)(v < 0 ? v - 0.5f : v + 0.5f);
      }
   }

   // Encode DC
   diff = DU[0] - DC;
   if (diff == 0) {
      stbiw__jpg_writeBits(s, bitBuf, bitCnt, HTDC[0]);
   } else {
      unsigned short bits[2];
      stbiw__jpg_calcBits(diff, bits);
      stbiw__jpg_writeBits(s, bitBuf, bitCnt, HTDC[bits[1]]);
      stbiw__jpg_writeBits(s, bitBuf, bitCnt, bits);
   }
   // Encode ACs
   end0pos = 63;
   for(; (end0pos>0)&&(DU[end0pos]==0); --end0pos) {
   }
   // end0pos = first element in reverse order !=0
   if(end0pos == 0) {
      stbiw__jpg_writeBits(s, bitBuf, bitCnt, EOB);
      return DU[0];
   }
   for(i = 1; i <= end0pos; ++i) {
      int startpos = i;
      int nrzeroes;
      unsigned short bits[2];
      for (; DU[i]==0 && i<=end0pos; ++i) {
      }
      nrzeroes = i-startpos;
      if ( nrzeroes >= 16 ) {
         int lng = nrzeroes>>4;
         int nrmarker;
         for (nrmarker=1; nrmarker <= lng; ++nrmarker)
            stbiw__jpg_writeBits(s, bitBuf, bitCnt, M16zeroes);
         nrzeroes &= 15;
      }
      stbiw__jpg_calcBits(DU[i], bits);
      stbiw__jpg_writeBits(s, bitBuf, bitCnt, HTAC[(nrzeroes<<4)+bits[1]]);
      stbiw__jpg_writeBits(s, bitBuf, bitCnt, bits);
   }
   if(end0pos != 63) {
      stbiw__jpg_writeBits(s, bitBuf, bitCnt, EOB);
   }
   return DU[0];
}

static int stbi_write_jpg_core(stbi__write_context *s, int width, int height, int comp, const void* data, int quality) {
   // Constants that don't pollute global namespace
   static const unsigned char std_dc_luminance_nrcodes[] = {0,0,1,5,1,1,1,1,1,1,0,0,0,0,0,0,0};
   static const unsigned char std_dc_luminance_values[] = {0,1,2,3,4,5,6,7,8,9,10,11};
   static const unsigned char std_ac_luminance_nrcodes[] = {0,0,2,1,3,3,2,4,3,5,5,4,4,0,0,1,0x7d};
   static const unsigned char std_ac_luminance_values[] = {
      0x01,0x02,0x03,0x00,0x04,0x11,0x05,0x12,0x21,0x31,0x41,0x06,0x13,0x51,0x61,0x07,0x22,0x71,0x14,0x32,0x81,0x91,0xa1,0x08,
      0x23,0x42,0xb1,0xc1,0x15,0x52,0xd1,0xf0,0x24,0x33,0x62,0x72,0x82,0x09,0x0a,0x16,0x17,0x18,0x19,0x1a,0x25,0x26,0x27,0x28,
      0x29,0x2a,0x34,0x35,0x36,0x37,0x38,0x39,0x3a,0x43,0x44,0x45,0x46,0x47,0x48,0x49,0x4a,0x53,0x54,0x55,0x56,0x57,0x58,0x59,
      0x5a,0x63,0x64,0x65,0x66,0x67,0x68,0x69,0x6a,0x73,0x74,0x75,0x76,0x77,0x78,0x79,0x7a,0x83,0x84,0x85,0x86,0x87,0x88,0x89,
      0x8a,0x92,0x93,0x94,0x95,0x96,0x97,0x98,0x99,0x9a,0xa2,0xa3,0xa4,0xa5,0xa6,0xa7,0xa8,0xa9,0xaa,0xb2,0xb3,0xb4,0xb5,0xb6,
      0xb7,0xb8,0xb9,0xba,0xc2,0xc3,0xc4,0xc5,0xc6,0xc7,0xc8,0xc9,0xca,0xd2,0xd3,0xd4,0xd5,0xd6,0xd7,0xd8,0xd9,0xda,0xe1,0xe2,
      0xe3,0xe4,0xe5,0xe6,0xe7,0xe8,0xe9,0xea,0xf1,0xf2,0xf3,0xf4,0xf5,0xf6,0xf7,0xf8,0xf9,0xfa
   };
   static const unsigned char std_dc_chrominance_nrcodes[] = {0,0,3,1,1,1,1,1,1,1,1,1,0,0,0,0,0};
   static const unsigned char std_dc_chrominance_values[] = {0,1,2,3,4,5,6,7,8,9,10,11};
   static const unsigned char std_ac_chrominance_nrcodes[] = {0,0,2,1,2,4,4,3,4,7,5,4,4,0,1,2,0x77};
   static const unsigned char std_ac_chrominance_values[] = {
      0x00,0x01,0x02,0x03,0x11,0x04,0x05,0x21,0x31,0x06,0x12,0x41,0x51,0x07,0x61,0x71,0x13,0x22,0x32,0x81,0x08,0x14,0x42,0x91,
      0xa1,0xb1,0xc1,0x09,0x23,0x33,0x52,0xf0,0x15,0x62,0x72,0xd1,0x0a,0x16,0x24,0x34,0xe1,0x25,0xf1,0x17,0x18,0x19,0x1a,0x26,
      0x27,0x28,0x29,0x2a,0x35,0x36,0x37,0x38,0x39,0x3a,0x43,0x44,0x45,0x46,0x47,0x48,0x49,0x4a,0x53,0x54,0x55,0x56,0x57,0x58,
      0x59,0x5a,0x63,0x64,0x65,0x66,0x67,0x68,0x69,0x6a,0x73,0x74,0x75,0x76,0x77,0x78,0x79,0x7a,0x82,0x83,0x84,0x85,0x86,0x87,
      0x88,0x89,0x8a,0x92,0x93,0x94,0x95,0x96,0x97,0x98,0x99,0x9a,0xa2,0xa3,0xa4,0xa5,0xa6,0xa7,0xa8,0xa9,0xaa,0xb2,0xb3,0xb4,
      0xb5,0xb6,0xb7,0xb8,0xb9,0xba,0xc2,0xc3,0xc4,0xc5,0xc6,0xc7,0xc8,0xc9,0xca,0xd2,0xd3,0xd4,0xd5,0xd6,0xd7,0xd8,0xd9,0xda,
      0xe2,0xe3,0xe4,0xe5,0xe6,0xe7,0xe8,0xe9,0xea,0xf2,0xf3,0xf4,0xf5,0xf6,0xf7,0xf8,0xf9,0xfa
   };
   // Huffman tables
   static const unsigned short YDC_HT[256][2] = { {0,2},{2,3},{3,3},{4,3},{5,3},{6,3},{14,4},{30,5},{62,6},{126,7},{254,8},{510,9}};
   static const unsigned short UVDC_HT[256][2] = { {0,2},{1,2},{2,2},{6,3},{14,4},{30,5},{62,6},{126,7},{254,8},{510,9},{1022,10},{2046,11}};
   static const unsigned short YAC_HT[256][2] = {
      {10,4},{0,2},{1,2},{4,3},{11,4},{26,5},{120,7},{248,8},{1014,10},{65410,16},{65411,16},{0,0},{0,0},{0,0},{0,0},{0,0},{0,0},
      {12,4},{27,5},{121,7},{502,9},{2038,11},{65412,16},{65413,16},{65414,16},{65415,16},{65416,16},{0,0},{0,0},{0,0},{0,0},{0,0},{0,0},
      {28,5},{249,8},{1015,10},{4084,12},{65417,16},{65418,16},{65419,16},{65420,16},{65421,16},{65422,16},{0,0},{0,0},{0,0},{0,0},{0,0},{0,0},
      {58,6},{503,9},{4085,12},{65423,16},{65424,16},{65425,16},{65426,16},{65427,16},{65428,16},{65429,16},{0,0},{0,0},{0,0},{0,0},{0,0},{0,0},
      {59,6},{1016,10},{65430,16},{65431,16},{65432,16},{65433,16},{65434,16},{65435,16},{65436,16},{65437,16},{0,0},{0,0},{0,0},{0,0},{0,0},{0,0},
      {122,7},{2039,11},{65438,16},{65439,16},{65440,16},{65441,16},{65442,16},{65443,16},{65444,16},{65445,16},{0,0},{0,0},{0,0},{0,0},{0,0},{0,0},
      {123,7},{4086,12},{65446,16},{65447,16},{65448,16},{65449,16},{65450,16},{65451,16},{65452,16},{65453,16},{0,0},{0,0},{0,0},{0,0},{0,0},{0,0},
      {250,8},{4087,12},{65454,16},{65455,16},{65456,16},{65457,16},{65458,16},{65459,16},{65460,16},{65461,16},{0,0},{0,0},{0,0},{0,0},{0,0},{0,0},
      {504,9},{32704,15},{65462,16},{65463,16},{65464,16},{65465,16},{65466,16},{65467,16},{65468,16},{65469,16},{0,0},{0,0},{0,0},{0,0},{0,0},{0,0},
      {505,9},{65470,16},{65471,16},{65472,16},{65473,16},{65474,16},{65475,16},{65476,16},{65477,16},{65478,16},{0,0},{0,0},{0,0},{0,0},{0,0},{0,0},
      {506,9},{65479,16},{65480,16},{65481,16},{65482,16},{65483,16},{65484,16},{65485,16},{65486,16},{65487,16},{0,0},{0,0},{0,0},{0,0},{0,0},{0,0},
      {1017,10},{65488,16},{65489,16},{65490,16},{65491,16},{65492,16},{65493,16},{65494,16},{65495,16},{65496,16},{0,0},{0,0},{0,0},{0,0},{0,0},{0,0},
      {1018,10},{65497,16},{65498,16},{65499,16},{65500,16},{65501,16},{65502,16},{65503,16},{65504,16},{65505,16},{0,0},{0,0},{0,0},{0,0},{0,0},{0,0},
      {2040,11},{65506,16},{65507,16},{65508,16},{65509,16},{65510,16},{65511,16},{65512,16},{65513,16},{65514,16},{0,0},{0,0},{0,0},{0,0},{0,0},{0,0},
      {65515,16},{65516,16},{65517,16},{65518,16},{65519,16},{65520,16},{65521,16},{65522,16},{65523,16},{65524,16},{0,0},{0,0},{0,0},{0,0},{0,0},
      {2041,11},{65525,16},{65526,16},{65527,16},{65528,16},{65529,16},{65530,16},{65531,16},{65532,16},{65533,16},{65534,16},{0,0},{0,0},{0,0},{0,0},{0,0}
   };
   static const unsigned short UVAC_HT[256][2] = {
      {0,2},{1,2},{4,3},{10,4},{24,5},{25,5},{56,6},{120,7},{500,9},{1014,10},{4084,12},{0,0},{0,0},{0,0},{0,0},{0,0},{0,0},
      {11,4},{57,6},{246,8},{501,9},{2038,11},{4085,12},{65416,16},{65417,16},{65418,16},{65419,16},{0,0},{0,0},{0,0},{0,0},{0,0},{0,0},
      {26,5},{247,8},{1015,10},{4086,12},{32706,15},{65420,16},{65421,16},{65422,16},{65423,16},{65424,16},{0,0},{0,0},{0,0},{0,0},{0,0},{0,0},
      {27,5},{248,8},{1016,10},{4087,12},{65425,16},{65426,16},{65427,16},{65428,16},{65429,16},{65430,16},{0,0},{0,0},{0,0},{0,0},{0,0},{0,0},
      {58,6},{502,9},{65431,16},{65432,16},{65433,16},{65434,16},{65435,16},{65436,16},{65437,16},{65438,16},{0,0},{0,0},{0,0},{0,0},{0,0},{0,0},
      {59,6},{1017,10},{65439,16},{65440,16},{65441,16},{65442,16},{65443,16},{65444,16},{65445,16},{65446,16},{0,0},{0,0},{0,0},{0,0},{0,0},{0,0},
      {121,7},{2039,11},{65447,16},{65448,16},{65449,16},{65450,16},{65451,16},{65452,16},{65453,16},{65454,16},{0,0},{0,0},{0,0},{0,0},{0,0},{0,0},
      {122,7},{2040,11},{65455,16},{65456,16},{65457,16},{65458,16},{65459,16},{65460,16},{65461,16},{65462,16},{0,0},{0,0},{0,0},{0,0},{0,0},{0,0},
      {249,8},{65463,16},{65464,16},{65465,16},{65466,16},{65467,16},{65468,16},{65469,16},{65470,16},{65471,16},{0,0},{0,0},{0,0},{0,0},{0,0},{0,0},
      {503,9},{65472,16},{65473,16},{65474,16},{65475,16},{65476,16},{65477,16},{65478,16},{65479,16},{65480,16},{0,0},{0,0},{0,0},{0,0},{0,0},{0,0},
      {504,9},{65481,16},{65482,16},{65483,16},{65484,16},{65485,16},{65486,16},{65487,16},{65488,16},{65489,16},{0,0},{0,0},{0,0},{0,0},{0,0},{0,0},
      {505,9},{65490,16},{65491,16},{65492,16},{65493,16},{65494,16},{65495,16},{65496,16},{65497,16},{65498,16},{0,0},{0,0},{0,0},{0,0},{0,0},{0,0},
      {506,9},{65499,16},{65500,16},{65501,16},{65502,16},{65503,16},{65504,16},{65505,16},{65506,16},{65507,16},{0,0},{0,0},{0,0},{0,0},{0,0},{0,0},
      {2041,11},{65508,16},{65509,16},{65510,16},{65511,16},{65512,16},{65513,16},{65514,16},{65515,16},{65516,16},{0,0},{0,0},{0,0},{0,0},{0,0},{0,0},
      {16352,14},{65517,16},{65518,16},{65519,16},{65520,16},{65521,16},{65522,16},{65523,16},{65524,16},{65525,16},{0,0},{0,0},{0,0},{0,0},{0,0},
      {1018,10},{32707,15},{65526,16},{65527,16},{65528,16},{65529,16},{65530,16},{65531,16},{65532,16},{65533,16},{65534,16},{0,0},{0,0},{0,0},{0,0},{0,0}
   };
   static const int YQT[] = {16,11,10,16,24,40,51,61,12,12,14,19,26,58,60,55,14,13,16,24,40,57,69,56,14,17,22,29,51,87,80,62,18,22,
                             37,56,68,109,103,77,24,35,55,64,81,104,113,92,49,64,78,87,103,121,120,101,72,92,95,98,112,100,103,99};
   static const int UVQT[] = {17,18,24,47,99,99,99,99,18,21,26,66,99,99,99,99,24,26,56,99,99,99,99,99,47,66,99,99,99,99,99,99,
                              99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99};
   static const float aasf[] = { 1.0f * 2.828427125f, 1.387039845f * 2.828427125f, 1.306562965f * 2.828427125f, 1.175875602f * 2.828427125f,
                                 1.0f * 2.828427125f, 0.785694958f * 2.828427125f, 0.541196100f * 2.828427125f, 0.275899379f * 2.828427125f };

   int row, col, i, k, subsample;
   float fdtbl_Y[64], fdtbl_UV[64];
   unsigned char YTable[64], UVTable[64];

   if(!data || !width || !height || comp > 4 || comp < 1) {
      return 0;
   }

   quality = quality ? quality : 90;
   subsample = quality <= 90 ? 1 : 0;
   quality = quality < 1 ? 1 : quality > 100 ? 100 : quality;
   quality = quality < 50 ? 5000 / quality : 200 - quality * 2;

   for(i = 0; i < 64; ++i) {
      int uvti, yti = (YQT[i]*quality+50)/100;
      YTable[stbiw__jpg_ZigZag[i]] = (unsigned char) (yti < 1 ? 1 : yti > 255 ? 255 : yti);
      uvti = (UVQT[i]*quality+50)/100;
      UVTable[stbiw__jpg_ZigZag[i]] = (unsigned char) (uvti < 1 ? 1 : uvti > 255 ? 255 : uvti);
   }

   for(row = 0, k = 0; row < 8; ++row) {
      for(col = 0; col < 8; ++col, ++k) {
         fdtbl_Y[k]  = 1 / (YTable [stbiw__jpg_ZigZag[k]] * aasf[row] * aasf[col]);
         fdtbl_UV[k] = 1 / (UVTable[stbiw__jpg_ZigZag[k]] * aasf[row] * aasf[col]);
      }
   }

   // Write Headers
   {
      static const unsigned char head0[] = { 0xFF,0xD8,0xFF,0xE0,0,0x10,'J','F','I','F',0,1,1,0,0,1,0,1,0,0,0xFF,0xDB,0,0x84,0 };
      static const unsigned char head2[] = { 0xFF,0xDA,0,0xC,3,1,0,2,0x11,3,0x11,0,0x3F,0 };
      const unsigned char head1[] = { 0xFF,0xC0,0,0x11,8,(unsigned char)(height>>8),STBIW_UCHAR(height),(unsigned char)(width>>8),STBIW_UCHAR(width),
                                      3,1,(unsigned char)(subsample?0x22:0x11),0,2,0x11,1,3,0x11,1,0xFF,0xC4,0x01,0xA2,0 };
      s->func(s->context, (void*)head0, sizeof(head0));
      s->func(s->context, (void*)YTable, sizeof(YTable));
      stbiw__putc(s, 1);
      s->func(s->context, UVTable, sizeof(UVTable));
      s->func(s->context, (void*)head1, sizeof(head1));
      s->func(s->context, (void*)(std_dc_luminance_nrcodes+1), sizeof(std_dc_luminance_nrcodes)-1);
      s->func(s->context, (void*)std_dc_luminance_values, sizeof(std_dc_luminance_values));
      stbiw__putc(s, 0x10); // HTYACinfo
      s->func(s->context, (void*)(std_ac_luminance_nrcodes+1), sizeof(std_ac_luminance_nrcodes)-1);
      s->func(s->context, (void*)std_ac_luminance_values, sizeof(std_ac_luminance_values));
      stbiw__putc(s, 1); // HTUDCinfo
      s->func(s->context, (void*)(std_dc_chrominance_nrcodes+1), sizeof(std_dc_chrominance_nrcodes)-1);
      s->func(s->context, (void*)std_dc_chrominance_values, sizeof(std_dc_chrominance_values));
      stbiw__putc(s, 0x11); // HTUACinfo
      s->func(s->context, (void*)(std_ac_chrominance_nrcodes+1), sizeof(std_ac_chrominance_nrcodes)-1);
      s->func(s->context, (void*)std_ac_chrominance_values, sizeof(std_ac_chrominance_values));
      s->func(s->context, (void*)head2, sizeof(head2));
   }

   // Encode 8x8 macroblocks
   {
      static const unsigned short fillBits[] = {0x7F, 7};
      int DCY=0, DCU=0, DCV=0;
      int bitBuf=0, bitCnt=0;
      // comp == 2 is grey+alpha (alpha is ignored)
      int ofsG = comp > 2 ? 1 : 0, ofsB = comp > 2 ? 2 : 0;
      const unsigned char *dataR = (const unsigned char *)data;
      const unsigned char *dataG = dataR + ofsG;
      const unsigned char *dataB = dataR + ofsB;
      int x, y, pos;
      if(subsample) {
         for(y = 0; y < height; y += 16) {
            for(x = 0; x < width; x += 16) {
               float Y[256], U[256], V[256];
               for(row = y, pos = 0; row < y+16; ++row) {
                  // row >= height => use last input row
                  int clamped_row = (row < height) ? row : height - 1;
                  int base_p = (stbi__flip_vertically_on_write ? (height-1-clamped_row) : clamped_row)*width*comp;
                  for(col = x; col < x+16; ++col, ++pos) {
                     // if col >= width => use pixel from last input column
                     int p = base_p + ((col < width) ? col : (width-1))*comp;
                     float r = dataR[p], g = dataG[p], b = dataB[p];
                     Y[pos]= +0.29900f*r + 0.58700f*g + 0.11400f*b - 128;
                     U[pos]= -0.16874f*r - 0.33126f*g + 0.50000f*b;
                     V[pos]= +0.50000f*r - 0.41869f*g - 0.08131f*b;
                  }
               }
               DCY = stbiw__jpg_processDU(s, &bitBuf, &bitCnt, Y+0,   16, fdtbl_Y, DCY, YDC_HT, YAC_HT);
               DCY = stbiw__jpg_processDU(s, &bitBuf, &bitCnt, Y+8,   16, fdtbl_Y, DCY, YDC_HT, YAC_HT);
               DCY = stbiw__jpg_processDU(s, &bitBuf, &bitCnt, Y+128, 16, fdtbl_Y, DCY, YDC_HT, YAC_HT);
               DCY = stbiw__jpg_processDU(s, &bitBuf, &bitCnt, Y+136, 16, fdtbl_Y, DCY, YDC_HT, YAC_HT);

               // subsample U,V
               {
                  float subU[64], subV[64];
                  int yy, xx;
                  for(yy = 0, pos = 0; yy < 8; ++yy) {
                     for(xx = 0; xx < 8; ++xx, ++pos) {
                        int j = yy*32+xx*2;
                        subU[pos] = (U[j+0] + U[j+1] + U[j+16] + U[j+17]) * 0.25f;
                        subV[pos] = (V[j+0] + V[j+1] + V[j+16] + V[j+17]) * 0.25f;
                     }
                  }
                  DCU = stbiw__jpg_processDU(s, &bitBuf, &bitCnt, subU, 8, fdtbl_UV, DCU, UVDC_HT, UVAC_HT);
                  DCV = stbiw__jpg_processDU(s, &bitBuf, &bitCnt, subV, 8, fdtbl_UV, DCV, UVDC_HT, UVAC_HT);
               }
            }
         }
      } else {
         for(y = 0; y < height; y += 8) {
            for(x = 0; x < width; x += 8) {
               float Y[64], U[64], V[64];
               for(row = y, pos = 0; row < y+8; ++row) {
                  // row >= height => use last input row
                  int clamped_row = (row < height) ? row : height - 1;
                  int base_p = (stbi__flip_vertically_on_write ? (height-1-clamped_row) : clamped_row)*width*comp;
                  for(col = x; col < x+8; ++col, ++pos) {
                     // if col >= width => use pixel from last input column
                     int p = base_p + ((col < width) ? col : (width-1))*comp;
                     float r = dataR[p], g = dataG[p], b = dataB[p];
                     Y[pos]= +0.29900f*r + 0.58700f*g + 0.11400f*b - 128;
                     U[pos]= -0.16874f*r - 0.33126f*g + 0.50000f*b;
                     V[pos]= +0.50000f*r - 0.41869f*g - 0.08131f*b;
                  }
               }

               DCY = stbiw__jpg_processDU(s, &bitBuf, &bitCnt, Y, 8, fdtbl_Y,  DCY, YDC_HT, YAC_HT);
               DCU = stbiw__jpg_processDU(s, &bitBuf, &bitCnt, U, 8, fdtbl_UV, DCU, UVDC_HT, UVAC_HT);
               DCV = stbiw__jpg_processDU(s, &bitBuf, &bitCnt, V, 8, fdtbl_UV, DCV, UVDC_HT, UVAC_HT);
            }
         }
      }

      // Do the bit alignment of the EOI marker
      stbiw__jpg_writeBits(s, &bitBuf, &bitCnt, fillBits);
   }

   // EOI
   stbiw__putc(s, 0xFF);
   stbiw__putc(s, 0xD9);

   return 1;
}

STBIWDEF int stbi_write_jpg_to_func(stbi_write_func *func, void *context, int x, int y, int comp, const void *data, int quality)
{
   stbi__write_context s = {0, 0, {0}, 0};
   stbi__start_write_callbacks(&s, func, context);
   return stbi_write_jpg_core(&s, x, y, comp, (void *) data, quality);
}


#ifndef STBI_WRITE_NO_STDIO
STBIWDEF int stbi_write_jpg(char const *filename, int x, int y, int comp, const void *data, int quality)
{
   stbi__write_context s = {0, 0, {0}, 0};
   if (stbi__start_write_file(&s,filename)) {
      int r = stbi_write_jpg_core(&s, x, y, comp, data, quality);
      stbi__end_write_file(&s);
      return r;
   } else
      return 0;
}
#endif

#endif
#include <vector>
#include <array>
#include <cassert>
#include <initializer_list>
#include <memory>
#include <stack>
template<typename T1, typename T2>
struct bigger_impl{
    using type = std::remove_all_extents_t<decltype(std::remove_all_extents_t<T1>{} + std::remove_all_extents_t<T2>{})>;
};
template<typename T1, typename T2>
using bigger = typename bigger_impl<T1, T2>::type;
template<typename T>
struct Vector2{
    using scalar = T;
    T x, y;
    
    #define OP2(X) Vector2<T> operator X(Vector2<T> o)const noexcept{return Vector2{x X o.x, y X o.y};}
    #define OPA2(X) Vector2<T>& operator X(Vector2<T> o)noexcept{x X o.x;y X o.y;return *this;}
    OP2(+)
    OP2(-)
    OP2(*)
    OP2(/)
    OPA2(+=)
    OPA2(-=)
    OPA2(*=)
    OPA2(/=)

    T operator[](size_t i)const noexcept{return reinterpret_cast<const T*>(this)[i];}
    T& operator[](size_t i)noexcept{return reinterpret_cast<T*>(this)[i];}
    Vector2<T> operator*(const T& o)const noexcept{return Vector2{x * o, y * o};}
    Vector2<T> operator+(const T& o)const noexcept{return Vector2{x + o, y + o};}
    Vector2<T> operator-()const noexcept{return Vector2<T>{-x, -y};}
    template<typename O>
    bigger<T, O> dot(const Vector2<O>& o)const noexcept{
        bigger<T, O> ret = x * o.x + y * o.y;
        return ret;
    }
    template<typename str>
    friend str& operator<<(str& s, const Vector2<T>& x){
        return s << x.x << ", " << x.y;
    }
    template<typename O>
    Vector2<O> cast()const noexcept{
        return Vector2<O>{O(x), O(y)};
    }
    Vector2<T> cwiseMin(const Vector2<T>& o)const noexcept{
        return Vector2<T>{std::min(x, o.x), std::min(y, o.y)};
    }
    Vector2<T> cwiseMax(const Vector2<T>& o)const noexcept{
        return Vector2<T>{std::max(x, o.x), std::max(y, o.y)};
    }
    T maxCoeff()const noexcept{
        return std::max(x, y);
    }
    T minCoeff()const noexcept{
        return std::min(x, y);
    }
};
template<typename T>
struct  Vector3{
    using scalar = T;
    T x, y, z;
    
    #define OP3(X) Vector3<T> operator X(Vector3<T> o)const noexcept{return Vector3{x X o.x, y X o.y, z X o.z};}
    #define OPA3(X) Vector3<T>& operator X(Vector3<T> o)noexcept{x X o.x;y X o.y;z X o.z;return *this;}
    OP3(+)
    OP3(-)
    OP3(*)
    OP3(/)
    OPA3(+=)
    OPA3(-=)
    OPA3(*=)
    OPA3(/=)

    T operator[](size_t i)const noexcept{return reinterpret_cast<const T*>(this)[i];}
    T& operator[](size_t i)noexcept{return reinterpret_cast<T*>(this)[i];}
    Vector3<T> operator*(const T& o)const noexcept{return Vector3{x * o, y * o, z * o};}
    Vector3<T> operator-()const noexcept{return Vector3<T>{-x, -y, -z};}
    template<typename O>
    Vector3<bigger<T, O>> cross(const Vector3<O>& o)const noexcept{
        return Vector3<bigger<T, O>>{y * o.z - z * o.y, z * o.x - x * o.z, x * o.y - y * o.x};
    }
    template<typename O>
    bigger<T, O> dot(const Vector3<O>& o)const noexcept{
        bigger<T, O> ret = x * o.x + y * o.y + z * o.z;
        return ret;
    }
    template<typename str>
    friend str& operator<<(str& s, const Vector3<T>& x){
        return s << x.x << ", " << x.y << ", " << x.z;
    }
    Vector3<T> cwiseMin(const Vector3<T>& o)const noexcept{
        return Vector3<T>{std::min(x, o.x), std::min(y, o.y), std::min(z, o.z)};
    }
    Vector3<T> cwiseMax(const Vector3<T>& o)const noexcept{
        return Vector3<T>{std::max(x, o.x), std::max(y, o.y), std::max(z, o.z)};
    }
    T maxCoeff()const noexcept{
        return std::max(std::max(x, y), z);
    }
    T minCoeff()const noexcept{
        return std::min(std::min(x, y), z);
    }
    template<typename O>
    Vector3<O> cast()const noexcept{
        return Vector3<O>{O(x), O(y), O(z)};
    }
};
template<typename T>
struct Vector4{
    using scalar = T;
    T x, y, z, w;
    #define OP4(X) Vector4<T> operator X(Vector4<T> o)const noexcept{return Vector4{x X o.x, y X o.y, z X o.z, w X o.w};}
    #define OPA4(X) Vector4<T>& operator X(Vector4<T> o)noexcept{x X o.x;y X o.y;z X o.z;w X o.w;return *this;}
    OP4(+)
    OP4(-)
    OP4(*)
    OP4(/)
    OPA4(+=)
    OPA4(-=)
    OPA4(*=)
    OPA4(/=)
    T operator[](size_t i)const noexcept{return reinterpret_cast<const T*>(this)[i];}
    T& operator[](size_t i)noexcept{return reinterpret_cast<T*>(this)[i];}
    Vector4<T> operator*(const T& o)const noexcept{return Vector4{x * o, y * o, z * o, w * o};}
    Vector4<T> operator-()const noexcept{return Vector4<T>{-x, -y, -z, -w};}
    template<typename O>
    bigger<T, O> dot(const Vector4<O>& o)const noexcept{
        bigger<T, O> ret = x * o.x + y * o.y + z * o.z + w * o.w;
        return ret;
    }
    template<typename str>
    friend str& operator<<(str& s, const Vector4<T>& x){
        return s << x.x << ", " << x.y << ", " << x.z << ", " << x.w;
    }
    Vector2<T> head2()const noexcept{
        return Vector2<T>{x, y};
    }
    Vector3<T> head3()const noexcept{
        return Vector3<T>{x, y, z};
    }
    Vector4<T> homogenize()const noexcept{
        T iw = T(1) / w;
        Vector4<T> ret(*this);
        ret.x *= iw;
        ret.y *= iw;
        ret.z *= iw;
        return ret;
    }
    template<typename O>
    Vector4<O> cast()const noexcept{
        return Vector4<O>{O(x), O(y), O(z), O(w)};
    }
};

template<typename T>
Vector2<T> operator*(T x, const Vector2<T>& v){
    return Vector2<T>{x * v.x, x * v.y};
}
template<typename T>
Vector3<T> operator*(T x, const Vector3<T>& v){
    return Vector3<T>{x * v.x, x * v.y, x * v.z};
}
template<typename T>
Vector4<T> operator*(T x, const Vector4<T>& v){
    return Vector4<T>{x * v.x, x * v.y, x * v.z, x * v.w};
}
template<typename T>
Vector4<T> zero_extend(const Vector3<T>& v){
    return Vector4<T>{v.x, v.y, v.z, T(0)};
}
template<typename T>
Vector4<T> one_extend(const Vector3<T>& o){
    Vector4<T> ret;
    ret.x = o.x;
    ret.y = o.y;
    ret.z = o.z;
    ret.w = T(1);
    return ret;
}
template<typename T>
Vector3<T> normalize(const Vector3<T>& v){
    using std::sqrt;

    T n = v.dot(v);
    T isv = 1.0 / sqrt(n);
    return v * isv;
}
template<typename T>
Vector4<T> normalize(const Vector4<T>& v){
    using std::sqrt;

    T n = v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
    T isv = 1.0 / sqrt(n);
    return v * isv;
}
template<typename T>
struct Matrix4{
    T data[16];
    Matrix4() : data{}{
        
    }
    Matrix4(const std::initializer_list<T>& t){
        assert(t.size() <= 16);
        std::copy(t.begin(), t.end(), data);
    }
    Matrix4(T x) : data{0}{
        data[0]  = x;
        data[5]  = x;
        data[10] = x;
        data[15] = x;
    }
    T operator()(size_t i, size_t j)const noexcept{
        return data[i + j * 4];
    }
    T& operator()(size_t i, size_t j) noexcept{
        return data[i + j * 4];
    }
    
    T operator[](size_t i)const noexcept{
        return data[i];
    }
    T& operator[](size_t i) noexcept{
        return data[i];
    }
    void setrow(size_t r, T x, T y, T z, T w){
        data[r + 0] = x;
        data[r + 4] = y;
        data[r + 8] = z;
        data[r + 12] = w;
    }
    template<typename str>
    friend str& operator<<(str& s, const Matrix4<T>& x){
        for(size_t i = 0;i < 4;i++){
            for(size_t j = 0;j < 4;j++){
                s << x(i, j) << ", ";
            }
            if(i < 3)
                s << "\n";
        }
        return s;
    }
    Matrix4<T> operator-()const noexcept{Matrix4<T> ret;for(size_t i = 0;i < 16;i++){ret[i] = -(this->operator[](i));}}
};
template<typename T, typename R>
Matrix4<bigger<T, R>> operator*(const Matrix4<T>& a, const Matrix4<R>& b){
    Matrix4<bigger<T, R>> ret(0);
    for(size_t i = 0;i < 4;i++){
        for(size_t j = 0;j < 4;j++){
            for(size_t k = 0;k < 4;k++){
                ret(i, j) += a(i, k) * b(k, j);
            }
        }
    }
    return ret;
}
template<typename T, typename R>
Vector4<bigger<T, R>> operator*(const Matrix4<T>& a, const Vector4<R>& b){
    Vector4<bigger<T, R>> ret{0,0,0,0};
    for(size_t i = 0;i < 4;i++){
        for(size_t j = 0;j < 4;j++){
            ret[j] += a(j, i) * b[i];
        }
    }
    return ret;
}
#ifdef __AVX__
#include <immintrin.h>

// Define a template specialization for float using AVX
template<>
inline Vector4<float> operator*(const Matrix4<float>& a, const Vector4<float>& b) {
    __m128 result = _mm_setzero_ps();
    for (size_t i = 0; i < 4; i++) {
        __m128 row = _mm_loadu_ps(&a.data[i * 4]);
        __m128 bValue = _mm_set1_ps(b[i]);
        result = _mm_fmadd_ps(row, bValue, result);
    }

    Vector4<float> ret;
    _mm_storeu_ps((float*)&ret, result);
    return ret;
}

// Define a template specialization for double using AVX
template<>
inline Vector4<double> operator*(const Matrix4<double>& a, const Vector4<double>& b) {
    __m256d result = _mm256_setzero_pd();
    for (size_t i = 0; i < 4; i++) {
        __m256d col = _mm256_loadu_pd(&a.data[i * 4]);
        __m256d bValue = _mm256_set1_pd(b[i]);
        result = _mm256_fmadd_pd(col, bValue, result);
    }

    Vector4<double> ret;
    _mm256_storeu_pd((double*)&ret, result);
    return ret;
}
#endif // __AVX__


template<typename T>
Matrix4<T> lookAt(Vector3<T> const& eye, Vector3<T> const& center, Vector3<T> const& up){
	const Vector3<T> f(normalize(center - eye));
	const Vector3<T> s(normalize(f.cross(up)));
	const Vector3<T> u(s.cross(f));
	Matrix4<T> Result(1);
	Result(0, 0) = s.x;
	Result(0, 1) = s.y;
	Result(0, 2) = s.z;
	Result(1, 0) = u.x;
	Result(1, 1) = u.y;
	Result(1, 2) = u.z;
	Result(2, 0) =-f.x;
	Result(2, 1) =-f.y;
	Result(2, 2) =-f.z;
	Result(0, 3) = -s.dot(eye);
	Result(1, 3) = -u.dot(eye);
	Result(2, 3) =  f.dot(eye);
	return Result;
}
template<typename T>
Matrix4<T> perspectiveRH_NO(T fovy, T aspect, T zNear, T zFar){
    using std::abs;
	assert(abs(aspect - std::numeric_limits<T>::epsilon()) > T(0));
	T const tanHalfFovy = tan(fovy / T(2));
	Matrix4<T> Result(T(0));
	Result(0, 0) = T(1) / (aspect * tanHalfFovy);
	Result(1, 1) = T(1) / (tanHalfFovy);
	Result(2, 2) = - (zFar + zNear) / (zFar - zNear);
	Result(3, 2) = - T(1);
	Result(2, 3) = - (T(2) * zFar * zNear) / (zFar - zNear);
	return Result;
}
template<typename T>
Matrix4<T> ortho(T left, T right, T bottom, T top, T zNear, T zFar){
	Matrix4<T> result(1);
	result(0, 0) = T(2) / (right - left);
	result(1, 1) = T(2) / (top - bottom);
	result(2, 2) = -T(2) / (zFar - zNear);
	result(0, 3) = -(right + left) / (right - left);
	result(1, 3) = -(top + bottom) / (top - bottom);
	result(2, 3) = -(zFar + zNear) / (zFar - zNear);
	return result;
}
struct camera{
    using vec3 = Vector3<float>;
    using mat4 = Matrix4<float>;
    vec3 pos;
    float pitch, yaw;
    camera(vec3 p, float pt, float y) : pos(p), pitch(pt), yaw(y){

    }
    camera(vec3 p, vec3 look) : pos(p){
        look = normalize(look);
        pitch = std::asin(look.y);
        yaw = std::atan2(look.z, look.x);
    }
    vec3 look_dir()const noexcept{
        vec3 fwd{std::cos(yaw) * std::cos(pitch), std::sin(pitch), std::sin(yaw) * std::cos(pitch)};
        return fwd;
    }
    vec3 left()const noexcept{
        vec3 fwd{std::cos(yaw) * std::cos(pitch), std::sin(pitch), std::sin(yaw) * std::cos(pitch)};
        vec3 up{0,1,0};
        return fwd.cross(up);
    }

    mat4 view_matrix()const noexcept{
        vec3 up       {0,1,0};
        vec3 fwd      {std::cos(yaw) * std::cos(pitch), std::sin(pitch), std::sin(yaw) * std::cos(pitch)};

        //[[maybe_unused]] vec3 realup = {fwd.cross(fwd.cross(up))};
        //[[maybe_unused]] vec3 right =  fwd.cross(realup);
        mat4 ret = lookAt(pos, pos + fwd, up);
        return ret;
    }
    mat4 perspective_matrix(float width, float height)const noexcept{
        return perspectiveRH_NO(1.0f, width / height, 0.1f, 200.0f);
    }
    mat4 matrix(float width, float height)const noexcept{
        return perspective_matrix(width, height) * view_matrix();
    }
};
template<typename T>
struct ptr : public std::unique_ptr<T>{
    using base = std::unique_ptr<T>;
    using base::base;
    ptr() : base(){}
    ptr(base b) : base(std::move(b)){}
    /*ptr(T* v) : base(v){

    }
    ptr(base&& v) : base(std::move(v)){

    }*/
    operator T*() noexcept{
        return base::get();
    }
    operator const T*() const noexcept{
        return base::get();
    }
    operator const unsigned char*() const noexcept{
        return (const unsigned char*)base::get();
    }
};
using Color = Vector4<unsigned char>;
template<typename _value_type>
struct basic_image {
    using value_type = _value_type;
    ptr<value_type[]> data;             // Image raw data
    unsigned int width;                 // Image base width
    unsigned int height;                // Image base height
    basic_image(unsigned w, unsigned h) : data(std::make_unique<value_type[]>(w * h)), width(w), height(h) {
        //data.reset(new value_type[w * h]);
    }
    basic_image(unsigned w, unsigned h, std::unique_ptr<value_type[]> rdata) : data(std::move(rdata)), width(w), height(h) {
    }
    const value_type& operator()(unsigned i, unsigned j)const noexcept{
        return data[i + j * width];
    }
    value_type& operator()(unsigned i, unsigned j)noexcept{
        return data[i + j * width];
    }
    //Unsupported options
    //int mipmaps;                // Mipmap levels, 1 by default
    //int format;                 // Data format (PixelFormat type)
};
using Image = basic_image<Color>;
struct framebuffer{
    using color_t = Vector3<float>;
    using depth_t = float;
    constexpr static depth_t empty_depth = depth_t(INFINITY);
    Vector2<unsigned int> resolution;
    Vector2<unsigned int> resolution_minus_one;
    
    basic_image<color_t> color_buffer;
    basic_image<depth_t> depth_buffer;

    Vector2<float> two_over_resolution;
    framebuffer(unsigned int w, unsigned int h) : resolution{w, h}, resolution_minus_one{w - 1, h - 1}, color_buffer(w, h), depth_buffer(w, h){
        assert(w && h && (w < 65356) && (h < 65356) && "Need positive and nonzero extents and reasonably big");
        two_over_resolution.x = 2.0f / resolution.x;
        two_over_resolution.y = 2.0f / resolution.y;
        std::fill(depth_buffer.data.get(), depth_buffer.data.get() + w * h, empty_depth);
    }
    void paint_pixeli(unsigned i, unsigned j, const color_t& color, float alpha, float depth){
        if(i >= resolution.x || j >= resolution.y){
            return;
        }
        //std::cout << i << " " << j << std::endl;
        //std::cout << color << std::endl;
        //std::cout << resolution.y << std::endl;
        if(depth_buffer(i, j) <= depth){
            return;
        }
        
        depth_buffer(i, j) = depth;
        color_t prevc = color_buffer(i, j);
        color_buffer(i, j) = color * alpha + prevc * (1.0f - alpha);
    }
    void paint_pixel(float x, float y, const color_t& color, float alpha, float depth){
        float xnrm = (x + 1) * 0.5f * float(resolution.x);
        float ynrm = (y + 1) * 0.5f * float(resolution.y);
        paint_pixeli((unsigned)xnrm, (unsigned)ynrm, color, alpha, depth);
    }
    Vector2<int> clip2screen(Vector2<float> x)const noexcept{
        x.y = -x.y;
        return ((x * 0.5f + 0.5f) * resolution.cast<float>()).cast<int>();
    }
    Vector2<float> screen2clip(Vector2<int> c)const noexcept{
        c.y = resolution_minus_one.y - c.y;
        return ((c.cast<float>() * two_over_resolution) + -1.0f);
    }
};
typedef struct Mesh {
    int vertexCount;        // Number of vertices stored in arrays
    int triangleCount;      // Number of triangles stored (indexed or not)

    // Vertex attributes data
    float *vertices;        // Vertex position (XYZ - 3 components per vertex) (shader-location = 0)
    float *texcoords;       // Vertex texture coordinates (UV - 2 components per vertex) (shader-location = 1)
    //float *texcoords2;      // Vertex texture second coordinates (UV - 2 components per vertex) (shader-location = 5)
    //float *normals;         // Vertex normals (XYZ - 3 components per vertex) (shader-location = 2)
    //float *tangents;        // Vertex tangents (XYZW - 4 components per vertex) (shader-location = 4)
    unsigned char *colors;      // Vertex colors (RGBA - 4 components per vertex) (shader-location = 3)
    unsigned short *indices;    // Vertex indices (in case vertex data comes indexed)
} Mesh;
Mesh GenMeshSphere(float radius, int rings, int slices);
template<typename _scalar>
struct barycentric_triangle_function{
    using scalar = _scalar;
    using vec2 = Vector2<_scalar>;
    using vec3 = Vector3<_scalar>;
    using vec4 = Vector4<_scalar>;
    std::array<vec4, 3> vertices;
    vec3 one_over_ws;
    
    scalar inv_detT;

    barycentric_triangle_function(const vec4& v1, const vec4& v2, const vec4& v3){
        using std::abs;
        vec2 T[2];
        vertices[0] = v1;
        vertices[1] = v2;
        vertices[2] = v3;
        one_over_ws = vec3{scalar(1) / v1.w, scalar(1) / v2.w, scalar(1) / v3.w};
        T[0] = vertices[1].head2() - vertices[0].head2();
        T[1] = vertices[2].head2() - vertices[0].head2();
        inv_detT = (scalar(1.0) / (T[0].x * T[1].y - T[0].y * T[1].x));
    }
    template<typename attribute>
    attribute perspective_correct(const vec2& p, attribute av1, attribute av2, attribute av3){
        vec3 lin = linear(p);
        vec3 one_over_w = one_over_ws.cwiseProduct(lin);
        attribute ret;
        ret = lin[0] * av1 * one_over_ws[0] + lin[1] * av2 * one_over_ws[1] + lin[2] * av3 * one_over_ws[2];
        return ret / one_over_w.sum();
    }
    template<typename attribute>
    attribute perspective_correct(const vec3& lin, const vec2&, attribute av1, attribute av2, attribute av3){
        vec3 one_over_w = one_over_ws.cwiseProduct(lin);
        attribute ret;
        ret = lin[0] * av1 * one_over_ws[0] + lin[1] * av2 * one_over_ws[1] + lin[2] * av3 * one_over_ws[2];
        return ret / one_over_w.sum();
    }
    template<typename attribute>
    attribute perspective_correct2(const vec3& lin, const vec3& /*one_over_w*/, _scalar isum, const vec2& /*p*/, attribute av1, attribute av2, attribute av3){
        attribute ret;
        ret = lin[0] * av1 * one_over_ws[0] + lin[1] * av2 * one_over_ws[1] + lin[2] * av3 * one_over_ws[2];
        return ret * isum;
    }
    vec3 linear(const vec2& p)const noexcept{
        scalar l1 = (vertices[1].y - vertices[2].y) * (p.x - vertices[2].x)
        + (vertices[2].x - vertices[1].x) * (p.y - vertices[2].y);

        scalar l2 = (vertices[2].y - vertices[0].y) * (p.x - vertices[2].x)
        + (vertices[0].x - vertices[2].x) * (p.y - vertices[2].y);
        l1 *= inv_detT;
        l2 *= inv_detT;
        vec3 ret{l1, l2, scalar(1) - l1 - l2};
       
        //ret = ret.cwiseMax(0.0f).cwiseMin(1.0f);
        return ret;
    }
};
struct vertex{
    using pos_t = Vector3<float>;
    using uv_t = Vector2<float>;
    using color_t = Vector3<float>;
    pos_t pos;
    uv_t uv;
    color_t color;
};
enum draw_mode{
    nothing, triangles
};



Image GenImageChecked(int width, int height, int checksX, int checksY, Color col1, Color col2);
Vector4<float> texture2D(const Image& img, const Vector2<float>& uv);
template<bool textured>
void draw_triangle_already_projected(framebuffer& img, vertex p1, vertex p2, vertex p3, const Image* texture = nullptr);
template<bool textured = false>
void draw_triangle(framebuffer& img, const Matrix4<float>& mat, vertex p1, vertex p2, vertex p3, const Image* texture = nullptr);
void depthblend_framebuffers(framebuffer& target, const framebuffer& op);
void rlBegin(draw_mode mode);
void rlVertex3f(float x, float y, float z);
void rlVertex2f(float x, float y);
void rlColor3f(float r, float g, float b);
void rlTexCoord2f(float r, float g);
void rlEnd();
void DrawMesh(Mesh mesh, Matrix4<float> transform);
void BeginTextureMode(framebuffer& fb);
void EndTextureMode();
void set_texture(Image* image);
void unset_texture();
void DrawTriangleStrip(const Vector2<float> *points, int pointCount, Color color);
void DrawBillboardLineEx(Vector3<float> startPos, Vector3<float> endPos, float thick, Color color);
void DrawLineEx(Vector2<float> startPos, Vector2<float> endPos, float thick, Color color);
void DrawRectangle(Vector2<float> pos, Vector2<float> ext);
void ClearBackground(Color col);
extern framebuffer* current_fb;
extern framebuffer* default_fb;
extern Image* active_texture;
extern draw_mode cmode;
extern vertex::uv_t current_uv;
extern vertex::color_t current_color;
extern std::vector<vertex> current_buffer;
extern std::stack<Matrix4<float>> matrix_stack;
void InitWindow(unsigned w, unsigned h);
void outputPPM(const framebuffer& fb, const std::string& filename);
void outputBMP(const framebuffer& fb, const std::string& filename);

#ifdef FRAST3D_IMPLEMENTATION
#define PAR_SHAPES_IMPLEMENTATION
#ifndef PAR_SHAPES_H
#define PAR_SHAPES_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>
// Ray (@raysan5): Commented to avoid conflict with raylib bool
/*
#if !defined(_MSC_VER)
# include <stdbool.h>
#else // MSVC
# if _MSC_VER >= 1800
#  include <stdbool.h>
# else // stdbool.h missing prior to MSVC++ 12.0 (VS2013)
#  define bool int
#  define true 1
#  define false 0
# endif
#endif
*/

#ifndef PAR_SHAPES_T
#define PAR_SHAPES_T uint16_t
#endif

typedef struct par_shapes_mesh_s {
    float* points;           // Flat list of 3-tuples (X Y Z X Y Z...)
    int npoints;             // Number of points
    PAR_SHAPES_T* triangles; // Flat list of 3-tuples (I J K I J K...)
    int ntriangles;          // Number of triangles
    float* normals;          // Optional list of 3-tuples (X Y Z X Y Z...)
    float* tcoords;          // Optional list of 2-tuples (U V U V U V...)
} par_shapes_mesh;

void par_shapes_free_mesh(par_shapes_mesh*);

// Generators ------------------------------------------------------------------

// Instance a cylinder that sits on the Z=0 plane using the given tessellation
// levels across the UV domain.  Think of "slices" like a number of pizza
// slices, and "stacks" like a number of stacked rings.  Height and radius are
// both 1.0, but they can easily be changed with par_shapes_scale.
par_shapes_mesh* par_shapes_create_cylinder(int slices, int stacks);

// Cone is similar to cylinder but the radius diminishes to zero as Z increases.
// Again, height and radius are 1.0, but can be changed with par_shapes_scale.
par_shapes_mesh* par_shapes_create_cone(int slices, int stacks);

// Create a disk of radius 1.0 with texture coordinates and normals by squashing
// a cone flat on the Z=0 plane.
par_shapes_mesh* par_shapes_create_parametric_disk(int slices, int stacks);

// Create a donut that sits on the Z=0 plane with the specified inner radius.
// The outer radius can be controlled with par_shapes_scale.
par_shapes_mesh* par_shapes_create_torus(int slices, int stacks, float radius);

// Create a sphere with texture coordinates and small triangles near the poles.
par_shapes_mesh* par_shapes_create_parametric_sphere(int slices, int stacks);

// Approximate a sphere with a subdivided icosahedron, which produces a nice
// distribution of triangles, but no texture coordinates.  Each subdivision
// level scales the number of triangles by four, so use a very low number.
par_shapes_mesh* par_shapes_create_subdivided_sphere(int nsubdivisions);

// More parametric surfaces.
par_shapes_mesh* par_shapes_create_klein_bottle(int slices, int stacks);
par_shapes_mesh* par_shapes_create_trefoil_knot(int slices, int stacks,
    float radius);
par_shapes_mesh* par_shapes_create_hemisphere(int slices, int stacks);
par_shapes_mesh* par_shapes_create_plane(int slices, int stacks);

// Create a parametric surface from a callback function that consumes a 2D
// point in [0,1] and produces a 3D point.
typedef void (*par_shapes_fn)(float const*, float*, void*);
par_shapes_mesh* par_shapes_create_parametric(par_shapes_fn, int slices,
    int stacks, void* userdata);

// Generate points for a 20-sided polyhedron that fits in the unit sphere.
// Texture coordinates and normals are not generated.
par_shapes_mesh* par_shapes_create_icosahedron();

// Generate points for a 12-sided polyhedron that fits in the unit sphere.
// Again, texture coordinates and normals are not generated.
par_shapes_mesh* par_shapes_create_dodecahedron();

// More platonic solids.
par_shapes_mesh* par_shapes_create_octahedron();
par_shapes_mesh* par_shapes_create_tetrahedron();
par_shapes_mesh* par_shapes_create_cube();

// Generate an orientable disk shape in 3-space.  Does not include normals or
// texture coordinates.
par_shapes_mesh* par_shapes_create_disk(float radius, int slices,
    float const* center, float const* normal);

// Create an empty shape.  Useful for building scenes with merge_and_free.
par_shapes_mesh* par_shapes_create_empty();

// Generate a rock shape that sits on the Y=0 plane, and sinks into it a bit.
// This includes smooth normals but no texture coordinates.  Each subdivision
// level scales the number of triangles by four, so use a very low number.
par_shapes_mesh* par_shapes_create_rock(int seed, int nsubdivisions);

// Create trees or vegetation by executing a recursive turtle graphics program.
// The program is a list of command-argument pairs.  See the unit test for
// an example.  Texture coordinates and normals are not generated.
par_shapes_mesh* par_shapes_create_lsystem(char const* program, int slices,
    int maxdepth);

// Queries ---------------------------------------------------------------------

// Dump out a text file conforming to the venerable OBJ format.
void par_shapes_export(par_shapes_mesh const*, char const* objfile);

// Take a pointer to 6 floats and set them to min xyz, max xyz.
void par_shapes_compute_aabb(par_shapes_mesh const* mesh, float* aabb);

// Make a deep copy of a mesh.  To make a brand new copy, pass null to "target".
// To avoid memory churn, pass an existing mesh to "target".
par_shapes_mesh* par_shapes_clone(par_shapes_mesh const* mesh,
    par_shapes_mesh* target);

// Transformations -------------------------------------------------------------

void par_shapes_merge(par_shapes_mesh* dst, par_shapes_mesh const* src);
void par_shapes_translate(par_shapes_mesh*, float x, float y, float z);
void par_shapes_rotate(par_shapes_mesh*, float radians, float const* axis);
void par_shapes_scale(par_shapes_mesh*, float x, float y, float z);
void par_shapes_merge_and_free(par_shapes_mesh* dst, par_shapes_mesh* src);

// Reverse the winding of a run of faces.  Useful when drawing the inside of
// a Cornell Box.  Pass 0 for nfaces to reverse every face in the mesh.
void par_shapes_invert(par_shapes_mesh*, int startface, int nfaces);

// Remove all triangles whose area is less than minarea.
void par_shapes_remove_degenerate(par_shapes_mesh*, float minarea);

// Dereference the entire index buffer and replace the point list.
// This creates an inefficient structure, but is useful for drawing facets.
// If create_indices is true, a trivial "0 1 2 3..." index buffer is generated.
void par_shapes_unweld(par_shapes_mesh* mesh, bool create_indices);

// Merge colocated verts, build a new index buffer, and return the
// optimized mesh.  Epsilon is the maximum distance to consider when
// welding vertices. The mapping argument can be null, or a pointer to
// npoints integers, which gets filled with the mapping from old vertex
// indices to new indices.
par_shapes_mesh* par_shapes_weld(par_shapes_mesh const*, float epsilon,
    PAR_SHAPES_T* mapping);

// Compute smooth normals by averaging adjacent facet normals.
void par_shapes_compute_normals(par_shapes_mesh* m);

// Global Config ---------------------------------------------------------------

void par_shapes_set_epsilon_welded_normals(float epsilon);
void par_shapes_set_epsilon_degenerate_sphere(float epsilon);

// Advanced --------------------------------------------------------------------

void par_shapes__compute_welded_normals(par_shapes_mesh* m);
void par_shapes__connect(par_shapes_mesh* scene, par_shapes_mesh* cylinder,
    int slices);

#ifndef PAR_PI
#define PAR_PI (3.14159265359)
#define PAR_MIN(a, b) (a > b ? b : a)
#define PAR_MAX(a, b) (a > b ? a : b)
#define PAR_CLAMP(v, lo, hi) PAR_MAX(lo, PAR_MIN(hi, v))
#define PAR_SWAP(T, A, B) { T tmp = B; B = A; A = tmp; }
#define PAR_SQR(a) ((a) * (a))
#endif

#ifndef PAR_MALLOC
#define PAR_MALLOC(T, N) ((T*) malloc(N * sizeof(T)))
#define PAR_CALLOC(T, N) ((T*) calloc(N * sizeof(T), 1))
#define PAR_REALLOC(T, BUF, N) ((T*) realloc(BUF, sizeof(T) * (N)))
#define PAR_FREE(BUF) free(BUF)
#endif

#ifdef __cplusplus
}
#endif

// -----------------------------------------------------------------------------
// END PUBLIC API
// -----------------------------------------------------------------------------

#ifdef PAR_SHAPES_IMPLEMENTATION
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <float.h>
#include <string.h>
#include <math.h>
#include <errno.h>

static float par_shapes__epsilon_welded_normals = 0.001;
static float par_shapes__epsilon_degenerate_sphere = 0.0001;

static void par_shapes__sphere(float const* uv, float* xyz, void*);
static void par_shapes__hemisphere(float const* uv, float* xyz, void*);
static void par_shapes__plane(float const* uv, float* xyz, void*);
static void par_shapes__klein(float const* uv, float* xyz, void*);
static void par_shapes__cylinder(float const* uv, float* xyz, void*);
static void par_shapes__cone(float const* uv, float* xyz, void*);
static void par_shapes__torus(float const* uv, float* xyz, void*);
static void par_shapes__trefoil(float const* uv, float* xyz, void*);

struct osn_context;
static int par__simplex_noise(int64_t seed, struct osn_context** ctx);
static void par__simplex_noise_free(struct osn_context* ctx);
static double par__simplex_noise2(struct osn_context* ctx, double x, double y);

static void par_shapes__copy3(float* result, float const* a)
{
    result[0] = a[0];
    result[1] = a[1];
    result[2] = a[2];
}

static float par_shapes__dot3(float const* a, float const* b)
{
    return b[0] * a[0] + b[1] * a[1] + b[2] * a[2];
}

static void par_shapes__transform3(float* p, float const* x, float const* y,
    float const* z)
{
    float px = par_shapes__dot3(p, x);
    float py = par_shapes__dot3(p, y);
    float pz = par_shapes__dot3(p, z);
    p[0] = px;
    p[1] = py;
    p[2] = pz;
}

static void par_shapes__cross3(float* result, float const* a, float const* b)
{
    float x = (a[1] * b[2]) - (a[2] * b[1]);
    float y = (a[2] * b[0]) - (a[0] * b[2]);
    float z = (a[0] * b[1]) - (a[1] * b[0]);
    result[0] = x;
    result[1] = y;
    result[2] = z;
}

static void par_shapes__mix3(float* d, float const* a, float const* b, float t)
{
    float x = b[0] * t + a[0] * (1 - t);
    float y = b[1] * t + a[1] * (1 - t);
    float z = b[2] * t + a[2] * (1 - t);
    d[0] = x;
    d[1] = y;
    d[2] = z;
}

static void par_shapes__scale3(float* result, float a)
{
    result[0] *= a;
    result[1] *= a;
    result[2] *= a;
}

static void par_shapes__normalize3(float* v)
{
    float lsqr = sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]);
    if (lsqr > 0) {
        par_shapes__scale3(v, 1.0f / lsqr);
    }
}

static void par_shapes__subtract3(float* result, float const* a)
{
    result[0] -= a[0];
    result[1] -= a[1];
    result[2] -= a[2];
}

static void par_shapes__add3(float* result, float const* a)
{
    result[0] += a[0];
    result[1] += a[1];
    result[2] += a[2];
}

static float par_shapes__sqrdist3(float const* a, float const* b)
{
    float dx = a[0] - b[0];
    float dy = a[1] - b[1];
    float dz = a[2] - b[2];
    return dx * dx + dy * dy + dz * dz;
}

void par_shapes__compute_welded_normals(par_shapes_mesh* m)
{
    const float epsilon = par_shapes__epsilon_welded_normals;
    m->normals = PAR_MALLOC(float, m->npoints * 3);
    PAR_SHAPES_T* weldmap = PAR_MALLOC(PAR_SHAPES_T, m->npoints);
    par_shapes_mesh* welded = par_shapes_weld(m, epsilon, weldmap);
    par_shapes_compute_normals(welded);
    float* pdst = m->normals;
    for (int i = 0; i < m->npoints; i++, pdst += 3) {
        int d = weldmap[i];
        float const* pnormal = welded->normals + d * 3;
        pdst[0] = pnormal[0];
        pdst[1] = pnormal[1];
        pdst[2] = pnormal[2];
    }
    PAR_FREE(weldmap);
    par_shapes_free_mesh(welded);
}

par_shapes_mesh* par_shapes_create_cylinder(int slices, int stacks)
{
    if (slices < 3 || stacks < 1) {
        return 0;
    }
    return par_shapes_create_parametric(par_shapes__cylinder, slices,
        stacks, 0);
}

par_shapes_mesh* par_shapes_create_cone(int slices, int stacks)
{
    if (slices < 3 || stacks < 1) {
        return 0;
    }
    return par_shapes_create_parametric(par_shapes__cone, slices,
        stacks, 0);
}

par_shapes_mesh* par_shapes_create_parametric_disk(int slices, int stacks)
{
    par_shapes_mesh* m = par_shapes_create_cone(slices, stacks);
    if (m) {
        par_shapes_scale(m, 1.0f, 1.0f, 0.0f);
    }
    return m;
}

par_shapes_mesh* par_shapes_create_parametric_sphere(int slices, int stacks)
{
    if (slices < 3 || stacks < 3) {
        return 0;
    }
    par_shapes_mesh* m = par_shapes_create_parametric(par_shapes__sphere,
        slices, stacks, 0);
    par_shapes_remove_degenerate(m, par_shapes__epsilon_degenerate_sphere);
    return m;
}

par_shapes_mesh* par_shapes_create_hemisphere(int slices, int stacks)
{
    if (slices < 3 || stacks < 3) {
        return 0;
    }
    par_shapes_mesh* m = par_shapes_create_parametric(par_shapes__hemisphere,
        slices, stacks, 0);
    par_shapes_remove_degenerate(m, par_shapes__epsilon_degenerate_sphere);
    return m;
}

par_shapes_mesh* par_shapes_create_torus(int slices, int stacks, float radius)
{
    if (slices < 3 || stacks < 3) {
        return 0;
    }
    assert(radius <= 1.0 && "Use smaller radius to avoid self-intersection.");
    assert(radius >= 0.1 && "Use larger radius to avoid self-intersection.");
    void* userdata = (void*) &radius;
    return par_shapes_create_parametric(par_shapes__torus, slices,
        stacks, userdata);
}

par_shapes_mesh* par_shapes_create_klein_bottle(int slices, int stacks)
{
    if (slices < 3 || stacks < 3) {
        return 0;
    }
    par_shapes_mesh* mesh = par_shapes_create_parametric(
        par_shapes__klein, slices, stacks, 0);
    int face = 0;
    for (int stack = 0; stack < stacks; stack++) {
        for (int slice = 0; slice < slices; slice++, face += 2) {
            if (stack < 27 * stacks / 32) {
                par_shapes_invert(mesh, face, 2);
            }
        }
    }
    par_shapes__compute_welded_normals(mesh);
    return mesh;
}

par_shapes_mesh* par_shapes_create_trefoil_knot(int slices, int stacks,
    float radius)
{
    if (slices < 3 || stacks < 3) {
        return 0;
    }
    assert(radius <= 3.0 && "Use smaller radius to avoid self-intersection.");
    assert(radius >= 0.5 && "Use larger radius to avoid self-intersection.");
    void* userdata = (void*) &radius;
    return par_shapes_create_parametric(par_shapes__trefoil, slices,
        stacks, userdata);
}

par_shapes_mesh* par_shapes_create_plane(int slices, int stacks)
{
    if (slices < 1 || stacks < 1) {
        return 0;
    }
    return par_shapes_create_parametric(par_shapes__plane, slices,
        stacks, 0);
}

par_shapes_mesh* par_shapes_create_parametric(par_shapes_fn fn,
    int slices, int stacks, void* userdata)
{
    par_shapes_mesh* mesh = PAR_CALLOC(par_shapes_mesh, 1);

    // Generate verts.
    mesh->npoints = (slices + 1) * (stacks + 1);
    mesh->points = PAR_CALLOC(float, 3 * mesh->npoints);
    float uv[2];
    float xyz[3];
    float* points = mesh->points;
    for (int stack = 0; stack < stacks + 1; stack++) {
        uv[0] = (float) stack / stacks;
        for (int slice = 0; slice < slices + 1; slice++) {
            uv[1] = (float) slice / slices;
            fn(uv, xyz, userdata);
            *points++ = xyz[0];
            *points++ = xyz[1];
            *points++ = xyz[2];
        }
    }

    // Generate texture coordinates.
    mesh->tcoords = PAR_CALLOC(float, 2 * mesh->npoints);
    float* uvs = mesh->tcoords;
    for (int stack = 0; stack < stacks + 1; stack++) {
        uv[0] = (float) stack / stacks;
        for (int slice = 0; slice < slices + 1; slice++) {
            uv[1] = (float) slice / slices;
            *uvs++ = uv[0];
            *uvs++ = uv[1];
        }
    }

    // Generate faces.
    mesh->ntriangles = 2 * slices * stacks;
    mesh->triangles = PAR_CALLOC(PAR_SHAPES_T, 3 * mesh->ntriangles);
    int v = 0;
    PAR_SHAPES_T* face = mesh->triangles;
    for (int stack = 0; stack < stacks; stack++) {
        for (int slice = 0; slice < slices; slice++) {
            int next = slice + 1;
            *face++ = v + slice + slices + 1;
            *face++ = v + next;
            *face++ = v + slice;
            *face++ = v + slice + slices + 1;
            *face++ = v + next + slices + 1;
            *face++ = v + next;
        }
        v += slices + 1;
    }

    par_shapes__compute_welded_normals(mesh);
    return mesh;
}

void par_shapes_free_mesh(par_shapes_mesh* mesh)
{
    PAR_FREE(mesh->points);
    PAR_FREE(mesh->triangles);
    PAR_FREE(mesh->normals);
    PAR_FREE(mesh->tcoords);
    PAR_FREE(mesh);
}

void par_shapes_export(par_shapes_mesh const* mesh, char const* filename)
{
    FILE* objfile = fopen(filename, "wt");
    float const* points = mesh->points;
    float const* tcoords = mesh->tcoords;
    float const* norms = mesh->normals;
    PAR_SHAPES_T const* indices = mesh->triangles;
    if (tcoords && norms) {
        for (int nvert = 0; nvert < mesh->npoints; nvert++) {
            fprintf(objfile, "v %f %f %f\n", points[0], points[1], points[2]);
            fprintf(objfile, "vt %f %f\n", tcoords[0], tcoords[1]);
            fprintf(objfile, "vn %f %f %f\n", norms[0], norms[1], norms[2]);
            points += 3;
            norms += 3;
            tcoords += 2;
        }
        for (int nface = 0; nface < mesh->ntriangles; nface++) {
            int a = 1 + *indices++;
            int b = 1 + *indices++;
            int c = 1 + *indices++;
            fprintf(objfile, "f %d/%d/%d %d/%d/%d %d/%d/%d\n",
                a, a, a, b, b, b, c, c, c);
        }
    } else if (norms) {
        for (int nvert = 0; nvert < mesh->npoints; nvert++) {
            fprintf(objfile, "v %f %f %f\n", points[0], points[1], points[2]);
            fprintf(objfile, "vn %f %f %f\n", norms[0], norms[1], norms[2]);
            points += 3;
            norms += 3;
        }
        for (int nface = 0; nface < mesh->ntriangles; nface++) {
            int a = 1 + *indices++;
            int b = 1 + *indices++;
            int c = 1 + *indices++;
            fprintf(objfile, "f %d//%d %d//%d %d//%d\n", a, a, b, b, c, c);
        }
    } else if (tcoords) {
        for (int nvert = 0; nvert < mesh->npoints; nvert++) {
            fprintf(objfile, "v %f %f %f\n", points[0], points[1], points[2]);
            fprintf(objfile, "vt %f %f\n", tcoords[0], tcoords[1]);
            points += 3;
            tcoords += 2;
        }
        for (int nface = 0; nface < mesh->ntriangles; nface++) {
            int a = 1 + *indices++;
            int b = 1 + *indices++;
            int c = 1 + *indices++;
            fprintf(objfile, "f %d/%d %d/%d %d/%d\n", a, a, b, b, c, c);
        }
    } else {
        for (int nvert = 0; nvert < mesh->npoints; nvert++) {
            fprintf(objfile, "v %f %f %f\n", points[0], points[1], points[2]);
            points += 3;
        }
        for (int nface = 0; nface < mesh->ntriangles; nface++) {
            int a = 1 + *indices++;
            int b = 1 + *indices++;
            int c = 1 + *indices++;
            fprintf(objfile, "f %d %d %d\n", a, b, c);
        }
    }
    fclose(objfile);
}

static void par_shapes__sphere(float const* uv, float* xyz, void*)
{
    float phi = uv[0] * PAR_PI;
    float theta = uv[1] * 2 * PAR_PI;
    xyz[0] = cosf(theta) * sinf(phi);
    xyz[1] = sinf(theta) * sinf(phi);
    xyz[2] = cosf(phi);
}

static void par_shapes__hemisphere(float const* uv, float* xyz, void*)
{
    float phi = uv[0] * PAR_PI;
    float theta = uv[1] * PAR_PI;
    xyz[0] = cosf(theta) * sinf(phi);
    xyz[1] = sinf(theta) * sinf(phi);
    xyz[2] = cosf(phi);
}

static void par_shapes__plane(float const* uv, float* xyz, void*)
{
    xyz[0] = uv[0];
    xyz[1] = uv[1];
    xyz[2] = 0;
}

static void par_shapes__klein(float const* uv, float* xyz, void*)
{
    float u = uv[0] * PAR_PI;
    float v = uv[1] * 2 * PAR_PI;
    u = u * 2;
    if (u < PAR_PI) {
        xyz[0] = 3 * cosf(u) * (1 + sinf(u)) + (2 * (1 - cosf(u) / 2)) *
            cosf(u) * cosf(v);
        xyz[2] = -8 * sinf(u) - 2 * (1 - cosf(u) / 2) * sinf(u) * cosf(v);
    } else {
        xyz[0] = 3 * cosf(u) * (1 + sinf(u)) + (2 * (1 - cosf(u) / 2)) *
            cosf(v + PAR_PI);
        xyz[2] = -8 * sinf(u);
    }
    xyz[1] = -2 * (1 - cosf(u) / 2) * sinf(v);
}

static void par_shapes__cylinder(float const* uv, float* xyz, void*)
{
    float theta = uv[1] * 2 * PAR_PI;
    xyz[0] = sinf(theta);
    xyz[1] = cosf(theta);
    xyz[2] = uv[0];
}

static void par_shapes__cone(float const* uv, float* xyz, void*)
{
    float r = 1.0f - uv[0];
    float theta = uv[1] * 2 * PAR_PI;
    xyz[0] = r * sinf(theta);
    xyz[1] = r * cosf(theta);
    xyz[2] = uv[0];
}

static void par_shapes__torus(float const* uv, float* xyz, void* userdata)
{
    float major = 1;
    float minor = *((float*) userdata);
    float theta = uv[0] * 2 * PAR_PI;
    float phi = uv[1] * 2 * PAR_PI;
    float beta = major + minor * cosf(phi);
    xyz[0] = cosf(theta) * beta;
    xyz[1] = sinf(theta) * beta;
    xyz[2] = sinf(phi) * minor;
}

static void par_shapes__trefoil(float const* uv, float* xyz, void* userdata)
{
    float minor = *((float*) userdata);
    const float a = 0.5f;
    const float b = 0.3f;
    const float c = 0.5f;
    const float d = minor * 0.1f;
    const float u = (1 - uv[0]) * 4 * PAR_PI;
    const float v = uv[1] * 2 * PAR_PI;
    const float r = a + b * cos(1.5f * u);
    const float x = r * cos(u);
    const float y = r * sin(u);
    const float z = c * sin(1.5f * u);
    float q[3];
    q[0] =
        -1.5f * b * sin(1.5f * u) * cos(u) - (a + b * cos(1.5f * u)) * sin(u);
    q[1] =
        -1.5f * b * sin(1.5f * u) * sin(u) + (a + b * cos(1.5f * u)) * cos(u);
    q[2] = 1.5f * c * cos(1.5f * u);
    par_shapes__normalize3(q);
    float qvn[3] = {q[1], -q[0], 0};
    par_shapes__normalize3(qvn);
    float ww[3];
    par_shapes__cross3(ww, q, qvn);
    xyz[0] = x + d * (qvn[0] * cos(v) + ww[0] * sin(v));
    xyz[1] = y + d * (qvn[1] * cos(v) + ww[1] * sin(v));
    xyz[2] = z + d * ww[2] * sin(v);
}

void par_shapes_set_epsilon_welded_normals(float epsilon) {
    par_shapes__epsilon_welded_normals = epsilon;
}

void par_shapes_set_epsilon_degenerate_sphere(float epsilon) {
    par_shapes__epsilon_degenerate_sphere = epsilon;
}

void par_shapes_merge(par_shapes_mesh* dst, par_shapes_mesh const* src)
{
    PAR_SHAPES_T offset = dst->npoints;
    int npoints = dst->npoints + src->npoints;
    int vecsize = sizeof(float) * 3;
    dst->points = PAR_REALLOC(float, dst->points, 3 * npoints);
    memcpy(dst->points + 3 * dst->npoints, src->points, vecsize * src->npoints);
    dst->npoints = npoints;
    if (src->normals || dst->normals) {
        dst->normals = PAR_REALLOC(float, dst->normals, 3 * npoints);
        if (src->normals) {
            memcpy(dst->normals + 3 * offset, src->normals,
                vecsize * src->npoints);
        }
    }
    if (src->tcoords || dst->tcoords) {
        int uvsize = sizeof(float) * 2;
        dst->tcoords = PAR_REALLOC(float, dst->tcoords, 2 * npoints);
        if (src->tcoords) {
            memcpy(dst->tcoords + 2 * offset, src->tcoords,
                uvsize * src->npoints);
        }
    }
    int ntriangles = dst->ntriangles + src->ntriangles;
    dst->triangles = PAR_REALLOC(PAR_SHAPES_T, dst->triangles, 3 * ntriangles);
    PAR_SHAPES_T* ptriangles = dst->triangles + 3 * dst->ntriangles;
    PAR_SHAPES_T const* striangles = src->triangles;
    for (int i = 0; i < src->ntriangles; i++) {
        *ptriangles++ = offset + *striangles++;
        *ptriangles++ = offset + *striangles++;
        *ptriangles++ = offset + *striangles++;
    }
    dst->ntriangles = ntriangles;
}

par_shapes_mesh* par_shapes_create_disk(float radius, int slices,
    float const* center, float const* normal)
{
    par_shapes_mesh* mesh = PAR_CALLOC(par_shapes_mesh, 1);
    mesh->npoints = slices + 1;
    mesh->points = PAR_MALLOC(float, 3 * mesh->npoints);
    float* points = mesh->points;
    *points++ = 0;
    *points++ = 0;
    *points++ = 0;
    for (int i = 0; i < slices; i++) {
        float theta = i * PAR_PI * 2 / slices;
        *points++ = radius * cos(theta);
        *points++ = radius * sin(theta);
        *points++ = 0;
    }
    float nnormal[3] = {normal[0], normal[1], normal[2]};
    par_shapes__normalize3(nnormal);
    mesh->normals = PAR_MALLOC(float, 3 * mesh->npoints);
    float* norms = mesh->normals;
    for (int i = 0; i < mesh->npoints; i++) {
        *norms++ = nnormal[0];
        *norms++ = nnormal[1];
        *norms++ = nnormal[2];
    }
    mesh->ntriangles = slices;
    mesh->triangles = PAR_MALLOC(PAR_SHAPES_T, 3 * mesh->ntriangles);
    PAR_SHAPES_T* triangles = mesh->triangles;
    for (int i = 0; i < slices; i++) {
        *triangles++ = 0;
        *triangles++ = 1 + i;
        *triangles++ = 1 + (i + 1) % slices;
    }
    float k[3] = {0, 0, -1};
    float axis[3];
    par_shapes__cross3(axis, nnormal, k);
    par_shapes__normalize3(axis);
    par_shapes_rotate(mesh, acos(nnormal[2]), axis);
    par_shapes_translate(mesh, center[0], center[1], center[2]);
    return mesh;
}

par_shapes_mesh* par_shapes_create_empty()
{
    return PAR_CALLOC(par_shapes_mesh, 1);
}

void par_shapes_translate(par_shapes_mesh* m, float x, float y, float z)
{
    float* points = m->points;
    for (int i = 0; i < m->npoints; i++) {
        *points++ += x;
        *points++ += y;
        *points++ += z;
    }
}

void par_shapes_rotate(par_shapes_mesh* mesh, float radians, float const* axis)
{
    float s = sinf(radians);
    float c = cosf(radians);
    float x = axis[0];
    float y = axis[1];
    float z = axis[2];
    float xy = x * y;
    float yz = y * z;
    float zx = z * x;
    float oneMinusC = 1.0f - c;
    float col0[3] = {
        (((x * x) * oneMinusC) + c),
        ((xy * oneMinusC) + (z * s)), ((zx * oneMinusC) - (y * s))
    };
    float col1[3] = {
        ((xy * oneMinusC) - (z * s)),
        (((y * y) * oneMinusC) + c), ((yz * oneMinusC) + (x * s))
    };
    float col2[3] = {
        ((zx * oneMinusC) + (y * s)),
        ((yz * oneMinusC) - (x * s)), (((z * z) * oneMinusC) + c)
    };
    float* p = mesh->points;
    for (int i = 0; i < mesh->npoints; i++, p += 3) {
        float x = col0[0] * p[0] + col1[0] * p[1] + col2[0] * p[2];
        float y = col0[1] * p[0] + col1[1] * p[1] + col2[1] * p[2];
        float z = col0[2] * p[0] + col1[2] * p[1] + col2[2] * p[2];
        p[0] = x;
        p[1] = y;
        p[2] = z;
    }
    float* n = mesh->normals;
    if (n) {
        for (int i = 0; i < mesh->npoints; i++, n += 3) {
            float x = col0[0] * n[0] + col1[0] * n[1] + col2[0] * n[2];
            float y = col0[1] * n[0] + col1[1] * n[1] + col2[1] * n[2];
            float z = col0[2] * n[0] + col1[2] * n[1] + col2[2] * n[2];
            n[0] = x;
            n[1] = y;
            n[2] = z;
        }
    }
}

void par_shapes_scale(par_shapes_mesh* m, float x, float y, float z)
{
    float* points = m->points;
    for (int i = 0; i < m->npoints; i++) {
        *points++ *= x;
        *points++ *= y;
        *points++ *= z;
    }
    float* n = m->normals;
    if (n && !(x == y && y == z)) {
        bool x_zero = x == 0;
        bool y_zero = y == 0;
        bool z_zero = z == 0;
        if (!x_zero && !y_zero && !z_zero) {
            x = 1.0f / x;
            y = 1.0f / y;
            z = 1.0f / z;
        } else {
            x = x_zero && !y_zero && !z_zero;
            y = y_zero && !x_zero && !z_zero;
            z = z_zero && !x_zero && !y_zero;
        }
        for (int i = 0; i < m->npoints; i++, n += 3) {
            n[0] *= x;
            n[1] *= y;
            n[2] *= z;
            par_shapes__normalize3(n);
        }
    }
}

void par_shapes_merge_and_free(par_shapes_mesh* dst, par_shapes_mesh* src)
{
    par_shapes_merge(dst, src);
    par_shapes_free_mesh(src);
}

void par_shapes_compute_aabb(par_shapes_mesh const* m, float* aabb)
{
    float* points = m->points;
    aabb[0] = aabb[3] = points[0];
    aabb[1] = aabb[4] = points[1];
    aabb[2] = aabb[5] = points[2];
    points += 3;
    for (int i = 1; i < m->npoints; i++, points += 3) {
        aabb[0] = PAR_MIN(points[0], aabb[0]);
        aabb[1] = PAR_MIN(points[1], aabb[1]);
        aabb[2] = PAR_MIN(points[2], aabb[2]);
        aabb[3] = PAR_MAX(points[0], aabb[3]);
        aabb[4] = PAR_MAX(points[1], aabb[4]);
        aabb[5] = PAR_MAX(points[2], aabb[5]);
    }
}

void par_shapes_invert(par_shapes_mesh* m, int face, int nfaces)
{
    nfaces = nfaces ? nfaces : m->ntriangles;
    PAR_SHAPES_T* tri = m->triangles + face * 3;
    for (int i = 0; i < nfaces; i++) {
        PAR_SWAP(PAR_SHAPES_T, tri[0], tri[2]);
        tri += 3;
    }
}

par_shapes_mesh* par_shapes_create_icosahedron()
{
    static float verts[] = {
        0.000,  0.000,  1.000,
        0.894,  0.000,  0.447,
        0.276,  0.851,  0.447,
        -0.724,  0.526,  0.447,
        -0.724, -0.526,  0.447,
        0.276, -0.851,  0.447,
        0.724,  0.526, -0.447,
        -0.276,  0.851, -0.447,
        -0.894,  0.000, -0.447,
        -0.276, -0.851, -0.447,
        0.724, -0.526, -0.447,
        0.000,  0.000, -1.000
    };
    static PAR_SHAPES_T faces[] = {
        0,1,2,
        0,2,3,
        0,3,4,
        0,4,5,
        0,5,1,
        7,6,11,
        8,7,11,
        9,8,11,
        10,9,11,
        6,10,11,
        6,2,1,
        7,3,2,
        8,4,3,
        9,5,4,
        10,1,5,
        6,7,2,
        7,8,3,
        8,9,4,
        9,10,5,
        10,6,1
    };
    par_shapes_mesh* mesh = PAR_CALLOC(par_shapes_mesh, 1);
    mesh->npoints = sizeof(verts) / sizeof(verts[0]) / 3;
    mesh->points = PAR_MALLOC(float, sizeof(verts) / 4);
    memcpy(mesh->points, verts, sizeof(verts));
    mesh->ntriangles = sizeof(faces) / sizeof(faces[0]) / 3;
    mesh->triangles = PAR_MALLOC(PAR_SHAPES_T, sizeof(faces) / 2);
    memcpy(mesh->triangles, faces, sizeof(faces));
    return mesh;
}

par_shapes_mesh* par_shapes_create_dodecahedron()
{
    static float verts[20 * 3] = {
        0.607, 0.000, 0.795,
        0.188, 0.577, 0.795,
        -0.491, 0.357, 0.795,
        -0.491, -0.357, 0.795,
        0.188, -0.577, 0.795,
        0.982, 0.000, 0.188,
        0.304, 0.934, 0.188,
        -0.795, 0.577, 0.188,
        -0.795, -0.577, 0.188,
        0.304, -0.934, 0.188,
        0.795, 0.577, -0.188,
        -0.304, 0.934, -0.188,
        -0.982, 0.000, -0.188,
        -0.304, -0.934, -0.188,
        0.795, -0.577, -0.188,
        0.491, 0.357, -0.795,
        -0.188, 0.577, -0.795,
        -0.607, 0.000, -0.795,
        -0.188, -0.577, -0.795,
        0.491, -0.357, -0.795,
    };
    static PAR_SHAPES_T pentagons[12 * 5] = {
        0,1,2,3,4,
        5,10,6,1,0,
        6,11,7,2,1,
        7,12,8,3,2,
        8,13,9,4,3,
        9,14,5,0,4,
        15,16,11,6,10,
        16,17,12,7,11,
        17,18,13,8,12,
        18,19,14,9,13,
        19,15,10,5,14,
        19,18,17,16,15
    };
    int npentagons = sizeof(pentagons) / sizeof(pentagons[0]) / 5;
    par_shapes_mesh* mesh = PAR_CALLOC(par_shapes_mesh, 1);
    int ncorners = sizeof(verts) / sizeof(verts[0]) / 3;
    mesh->npoints = ncorners;
    mesh->points = PAR_MALLOC(float, mesh->npoints * 3);
    memcpy(mesh->points, verts, sizeof(verts));
    PAR_SHAPES_T const* pentagon = pentagons;
    mesh->ntriangles = npentagons * 3;
    mesh->triangles = PAR_MALLOC(PAR_SHAPES_T, mesh->ntriangles * 3);
    PAR_SHAPES_T* tris = mesh->triangles;
    for (int p = 0; p < npentagons; p++, pentagon += 5) {
        *tris++ = pentagon[0];
        *tris++ = pentagon[1];
        *tris++ = pentagon[2];
        *tris++ = pentagon[0];
        *tris++ = pentagon[2];
        *tris++ = pentagon[3];
        *tris++ = pentagon[0];
        *tris++ = pentagon[3];
        *tris++ = pentagon[4];
    }
    return mesh;
}

par_shapes_mesh* par_shapes_create_octahedron()
{
    static float verts[6 * 3] = {
        0.000, 0.000, 1.000,
        1.000, 0.000, 0.000,
        0.000, 1.000, 0.000,
        -1.000, 0.000, 0.000,
        0.000, -1.000, 0.000,
        0.000, 0.000, -1.000
    };
    static PAR_SHAPES_T triangles[8 * 3] = {
        0,1,2,
        0,2,3,
        0,3,4,
        0,4,1,
        2,1,5,
        3,2,5,
        4,3,5,
        1,4,5,
    };
    int ntris = sizeof(triangles) / sizeof(triangles[0]) / 3;
    par_shapes_mesh* mesh = PAR_CALLOC(par_shapes_mesh, 1);
    int ncorners = sizeof(verts) / sizeof(verts[0]) / 3;
    mesh->npoints = ncorners;
    mesh->points = PAR_MALLOC(float, mesh->npoints * 3);
    memcpy(mesh->points, verts, sizeof(verts));
    PAR_SHAPES_T const* triangle = triangles;
    mesh->ntriangles = ntris;
    mesh->triangles = PAR_MALLOC(PAR_SHAPES_T, mesh->ntriangles * 3);
    PAR_SHAPES_T* tris = mesh->triangles;
    for (int p = 0; p < ntris; p++) {
        *tris++ = *triangle++;
        *tris++ = *triangle++;
        *tris++ = *triangle++;
    }
    return mesh;
}

par_shapes_mesh* par_shapes_create_tetrahedron()
{
    static float verts[4 * 3] = {
        0.000, 1.333, 0,
        0.943, 0, 0,
        -0.471, 0, 0.816,
        -0.471, 0, -0.816,
    };
    static PAR_SHAPES_T triangles[4 * 3] = {
        2,1,0,
        3,2,0,
        1,3,0,
        1,2,3,
    };
    int ntris = sizeof(triangles) / sizeof(triangles[0]) / 3;
    par_shapes_mesh* mesh = PAR_CALLOC(par_shapes_mesh, 1);
    int ncorners = sizeof(verts) / sizeof(verts[0]) / 3;
    mesh->npoints = ncorners;
    mesh->points = PAR_MALLOC(float, mesh->npoints * 3);
    memcpy(mesh->points, verts, sizeof(verts));
    PAR_SHAPES_T const* triangle = triangles;
    mesh->ntriangles = ntris;
    mesh->triangles = PAR_MALLOC(PAR_SHAPES_T, mesh->ntriangles * 3);
    PAR_SHAPES_T* tris = mesh->triangles;
    for (int p = 0; p < ntris; p++) {
        *tris++ = *triangle++;
        *tris++ = *triangle++;
        *tris++ = *triangle++;
    }
    return mesh;
}

par_shapes_mesh* par_shapes_create_cube()
{
    static float verts[8 * 3] = {
        0, 0, 0, // 0
        0, 1, 0, // 1
        1, 1, 0, // 2
        1, 0, 0, // 3
        0, 0, 1, // 4
        0, 1, 1, // 5
        1, 1, 1, // 6
        1, 0, 1, // 7
    };
    static PAR_SHAPES_T quads[6 * 4] = {
        7,6,5,4, // front
        0,1,2,3, // back
        6,7,3,2, // right
        5,6,2,1, // top
        4,5,1,0, // left
        7,4,0,3, // bottom
    };
    int nquads = sizeof(quads) / sizeof(quads[0]) / 4;
    par_shapes_mesh* mesh = PAR_CALLOC(par_shapes_mesh, 1);
    int ncorners = sizeof(verts) / sizeof(verts[0]) / 3;
    mesh->npoints = ncorners;
    mesh->points = PAR_MALLOC(float, mesh->npoints * 3);
    memcpy(mesh->points, verts, sizeof(verts));
    PAR_SHAPES_T const* quad = quads;
    mesh->ntriangles = nquads * 2;
    mesh->triangles = PAR_MALLOC(PAR_SHAPES_T, mesh->ntriangles * 3);
    PAR_SHAPES_T* tris = mesh->triangles;
    for (int p = 0; p < nquads; p++, quad += 4) {
        *tris++ = quad[0];
        *tris++ = quad[1];
        *tris++ = quad[2];
        *tris++ = quad[2];
        *tris++ = quad[3];
        *tris++ = quad[0];
    }
    return mesh;
}

typedef struct {
    char* cmd;
    char* arg;
} par_shapes__command;

typedef struct {
    char const* name;
    int weight;
    int ncommands;
    par_shapes__command* commands;
} par_shapes__rule;

typedef struct {
    int pc;
    float position[3];
    float scale[3];
    par_shapes_mesh* orientation;
    par_shapes__rule* rule;
} par_shapes__stackframe;

static par_shapes__rule* par_shapes__pick_rule(const char* name,
    par_shapes__rule* rules, int nrules){
    par_shapes__rule* rule = 0;
    int total = 0;
    for (int i = 0; i < nrules; i++) {
        rule = rules + i;
        if (!strcmp(rule->name, name)) {
            total += rule->weight;
        }
    }
    float r = (float) (rand() / double(RAND_MAX));
    float t = 0;
    for (int i = 0; i < nrules; i++) {
        rule = rules + i;
        if (!strcmp(rule->name, name)) {
            t += (float) rule->weight / total;
            if (t >= r) {
                return rule;
            }
        }
    }
    return rule;
}

static par_shapes_mesh* par_shapes__create_turtle(){
    const float xaxis[] = {1, 0, 0};
    const float yaxis[] = {0, 1, 0};
    const float zaxis[] = {0, 0, 1};
    par_shapes_mesh* turtle = PAR_CALLOC(par_shapes_mesh, 1);
    turtle->npoints = 3;
    turtle->points = PAR_CALLOC(float, turtle->npoints * 3);
    par_shapes__copy3(turtle->points + 0, xaxis);
    par_shapes__copy3(turtle->points + 3, yaxis);
    par_shapes__copy3(turtle->points + 6, zaxis);
    return turtle;
}

static par_shapes_mesh* par_shapes__apply_turtle(par_shapes_mesh* mesh,
    par_shapes_mesh* turtle, float const* pos, float const* scale)
{
    par_shapes_mesh* m = par_shapes_clone(mesh, 0);
    for (int p = 0; p < m->npoints; p++) {
        float* pt = m->points + p * 3;
        pt[0] *= scale[0];
        pt[1] *= scale[1];
        pt[2] *= scale[2];
        par_shapes__transform3(pt,
            turtle->points + 0, turtle->points + 3, turtle->points + 6);
        pt[0] += pos[0];
        pt[1] += pos[1];
        pt[2] += pos[2];
    }
    return m;
}

void par_shapes__connect(par_shapes_mesh* scene, par_shapes_mesh* cylinder,
    int slices)
{
    int stacks = 1;
    int npoints = (slices + 1) * (stacks + 1);
    assert(scene->npoints >= npoints && "Cannot connect to empty scene.");

    // Create the new point list.
    npoints = scene->npoints + (slices + 1);
    float* points = PAR_MALLOC(float, npoints * 3);
    memcpy(points, scene->points, sizeof(float) * scene->npoints * 3);
    float* newpts = points + scene->npoints * 3;
    memcpy(newpts, cylinder->points + (slices + 1) * 3,
        sizeof(float) * (slices + 1) * 3);
    PAR_FREE(scene->points);
    scene->points = points;

    // Create the new triangle list.
    int ntriangles = scene->ntriangles + 2 * slices * stacks;
    PAR_SHAPES_T* triangles = PAR_MALLOC(PAR_SHAPES_T, ntriangles * 3);
    memcpy(triangles, scene->triangles,
        sizeof(PAR_SHAPES_T) * scene->ntriangles * 3);
    int v = scene->npoints - (slices + 1);
    PAR_SHAPES_T* face = triangles + scene->ntriangles * 3;
    for (int stack = 0; stack < stacks; stack++) {
        for (int slice = 0; slice < slices; slice++) {
            int next = slice + 1;
            *face++ = v + slice + slices + 1;
            *face++ = v + next;
            *face++ = v + slice;
            *face++ = v + slice + slices + 1;
            *face++ = v + next + slices + 1;
            *face++ = v + next;
        }
        v += slices + 1;
    }
    PAR_FREE(scene->triangles);
    scene->triangles = triangles;

    scene->npoints = npoints;
    scene->ntriangles = ntriangles;
}

par_shapes_mesh* par_shapes_create_lsystem(char const* text, int slices,
    int maxdepth)
{
    char* program;
    program = PAR_MALLOC(char, strlen(text) + 1);

    // The first pass counts the number of rules and commands.
    strcpy(program, text);
    char *cmd = strtok(program, " ");
    int nrules = 1;
    int ncommands = 0;
    while (cmd) {
        char *arg = strtok(0, " ");
        if (!arg) {
            puts("lsystem error: unexpected end of program.");
            break;
        }
        if (!strcmp(cmd, "rule")) {
            nrules++;
        } else {
            ncommands++;
        }
        cmd = strtok(0, " ");
    }

    // Allocate space.
    par_shapes__rule* rules = PAR_MALLOC(par_shapes__rule, nrules);
    par_shapes__command* commands = PAR_MALLOC(par_shapes__command, ncommands);

    // Initialize the entry rule.
    par_shapes__rule* current_rule = &rules[0];
    par_shapes__command* current_command = &commands[0];
    current_rule->name = "entry";
    current_rule->weight = 1;
    current_rule->ncommands = 0;
    current_rule->commands = current_command;

    // The second pass fills in the structures.
    strcpy(program, text);
    cmd = strtok(program, " ");
    while (cmd) {
        char *arg = strtok(0, " ");
        if (!strcmp(cmd, "rule")) {
            current_rule++;

            // Split the argument into a rule name and weight.
            char* dot = strchr(arg, '.');
            if (dot) {
                current_rule->weight = atoi(dot + 1);
                *dot = 0;
            } else {
                current_rule->weight = 1;
            }

            current_rule->name = arg;
            current_rule->ncommands = 0;
            current_rule->commands = current_command;
        } else {
            current_rule->ncommands++;
            current_command->cmd = cmd;
            current_command->arg = arg;
            current_command++;
        }
        cmd = strtok(0, " ");
    }

    // For testing purposes, dump out the parsed program.
    #ifdef TEST_PARSE
    for (int i = 0; i < nrules; i++) {
        par_shapes__rule rule = rules[i];
        printf("rule %s.%d\n", rule.name, rule.weight);
        for (int c = 0; c < rule.ncommands; c++) {
            par_shapes__command cmd = rule.commands[c];
            printf("\t%s %s\n", cmd.cmd, cmd.arg);
        }
    }
    #endif

    // Instantiate the aggregated shape and the template shapes.
    par_shapes_mesh* scene = PAR_CALLOC(par_shapes_mesh, 1);
    par_shapes_mesh* tube = par_shapes_create_cylinder(slices, 1);
    par_shapes_mesh* turtle = par_shapes__create_turtle();

    // We're not attempting to support texture coordinates and normals
    // with L-systems, so remove them from the template shape.
    PAR_FREE(tube->normals);
    PAR_FREE(tube->tcoords);
    tube->normals = 0;
    tube->tcoords = 0;

    const float xaxis[] = {1, 0, 0};
    const float yaxis[] = {0, 1, 0};
    const float zaxis[] = {0, 0, 1};
    const float units[] = {1, 1, 1};

    // Execute the L-system program until the stack size is 0.
    par_shapes__stackframe* stack =
        PAR_CALLOC(par_shapes__stackframe, maxdepth);
    int stackptr = 0;
    stack[0].orientation = turtle;
    stack[0].rule = &rules[0];
    par_shapes__copy3(stack[0].scale, units);
    while (stackptr >= 0) {
        par_shapes__stackframe* frame = &stack[stackptr];
        par_shapes__rule* rule = frame->rule;
        par_shapes_mesh* turtle = frame->orientation;
        float* position = frame->position;
        float* scale = frame->scale;
        if (frame->pc >= rule->ncommands) {
            par_shapes_free_mesh(turtle);
            stackptr--;
            continue;
        }

        par_shapes__command* cmd = rule->commands + (frame->pc++);
        #ifdef DUMP_TRACE
        printf("%5s %5s %5s:%d  %03d\n", cmd->cmd, cmd->arg, rule->name,
            frame->pc - 1, stackptr);
        #endif

        float value;
        if (!strcmp(cmd->cmd, "shape")) {
            par_shapes_mesh* m = par_shapes__apply_turtle(tube, turtle,
                position, scale);
            if (!strcmp(cmd->arg, "connect")) {
                par_shapes__connect(scene, m, slices);
            } else {
                par_shapes_merge(scene, m);
            }
            par_shapes_free_mesh(m);
        } else if (!strcmp(cmd->cmd, "call") && stackptr < maxdepth - 1) {
            rule = par_shapes__pick_rule(cmd->arg, rules, nrules);
            frame = &stack[++stackptr];
            frame->rule = rule;
            frame->orientation = par_shapes_clone(turtle, 0);
            frame->pc = 0;
            par_shapes__copy3(frame->scale, scale);
            par_shapes__copy3(frame->position, position);
            continue;
        } else {
            value = atof(cmd->arg);
            if (!strcmp(cmd->cmd, "rx")) {
                par_shapes_rotate(turtle, value * PAR_PI / 180.0, xaxis);
            } else if (!strcmp(cmd->cmd, "ry")) {
                par_shapes_rotate(turtle, value * PAR_PI / 180.0, yaxis);
            } else if (!strcmp(cmd->cmd, "rz")) {
                par_shapes_rotate(turtle, value * PAR_PI / 180.0, zaxis);
            } else if (!strcmp(cmd->cmd, "tx")) {
                float vec[3] = {value, 0, 0};
                float t[3] = {
                    par_shapes__dot3(turtle->points + 0, vec),
                    par_shapes__dot3(turtle->points + 3, vec),
                    par_shapes__dot3(turtle->points + 6, vec)
                };
                par_shapes__add3(position, t);
            } else if (!strcmp(cmd->cmd, "ty")) {
                float vec[3] = {0, value, 0};
                float t[3] = {
                    par_shapes__dot3(turtle->points + 0, vec),
                    par_shapes__dot3(turtle->points + 3, vec),
                    par_shapes__dot3(turtle->points + 6, vec)
                };
                par_shapes__add3(position, t);
            } else if (!strcmp(cmd->cmd, "tz")) {
                float vec[3] = {0, 0, value};
                float t[3] = {
                    par_shapes__dot3(turtle->points + 0, vec),
                    par_shapes__dot3(turtle->points + 3, vec),
                    par_shapes__dot3(turtle->points + 6, vec)
                };
                par_shapes__add3(position, t);
            } else if (!strcmp(cmd->cmd, "sx")) {
                scale[0] *= value;
            } else if (!strcmp(cmd->cmd, "sy")) {
                scale[1] *= value;
            } else if (!strcmp(cmd->cmd, "sz")) {
                scale[2] *= value;
            } else if (!strcmp(cmd->cmd, "sa")) {
                scale[0] *= value;
                scale[1] *= value;
                scale[2] *= value;
            }
        }
    }
    PAR_FREE(stack);
    PAR_FREE(program);
    PAR_FREE(rules);
    PAR_FREE(commands);
    return scene;
}

void par_shapes_unweld(par_shapes_mesh* mesh, bool create_indices)
{
    int npoints = mesh->ntriangles * 3;
    float* points = PAR_MALLOC(float, 3 * npoints);
    float* dst = points;
    PAR_SHAPES_T const* index = mesh->triangles;
    for (int i = 0; i < npoints; i++) {
        float const* src = mesh->points + 3 * (*index++);
        *dst++ = src[0];
        *dst++ = src[1];
        *dst++ = src[2];
    }
    PAR_FREE(mesh->points);
    mesh->points = points;
    mesh->npoints = npoints;
    if (create_indices) {
        PAR_SHAPES_T* tris = PAR_MALLOC(PAR_SHAPES_T, 3 * mesh->ntriangles);
        PAR_SHAPES_T* index = tris;
        for (int i = 0; i < mesh->ntriangles * 3; i++) {
            *index++ = i;
        }
        PAR_FREE(mesh->triangles);
        mesh->triangles = tris;
    }
}

void par_shapes_compute_normals(par_shapes_mesh* m)
{
    PAR_FREE(m->normals);
    m->normals = PAR_CALLOC(float, m->npoints * 3);
    PAR_SHAPES_T const* triangle = m->triangles;
    float next[3], prev[3], cp[3];
    for (int f = 0; f < m->ntriangles; f++, triangle += 3) {
        float const* pa = m->points + 3 * triangle[0];
        float const* pb = m->points + 3 * triangle[1];
        float const* pc = m->points + 3 * triangle[2];
        par_shapes__copy3(next, pb);
        par_shapes__subtract3(next, pa);
        par_shapes__copy3(prev, pc);
        par_shapes__subtract3(prev, pa);
        par_shapes__cross3(cp, next, prev);
        par_shapes__add3(m->normals + 3 * triangle[0], cp);
        par_shapes__copy3(next, pc);
        par_shapes__subtract3(next, pb);
        par_shapes__copy3(prev, pa);
        par_shapes__subtract3(prev, pb);
        par_shapes__cross3(cp, next, prev);
        par_shapes__add3(m->normals + 3 * triangle[1], cp);
        par_shapes__copy3(next, pa);
        par_shapes__subtract3(next, pc);
        par_shapes__copy3(prev, pb);
        par_shapes__subtract3(prev, pc);
        par_shapes__cross3(cp, next, prev);
        par_shapes__add3(m->normals + 3 * triangle[2], cp);
    }
    float* normal = m->normals;
    for (int p = 0; p < m->npoints; p++, normal += 3) {
        par_shapes__normalize3(normal);
    }
}

static void par_shapes__subdivide(par_shapes_mesh* mesh)
{
    assert(mesh->npoints == mesh->ntriangles * 3 && "Must be unwelded.");
    int ntriangles = mesh->ntriangles * 4;
    int npoints = ntriangles * 3;
    float* points = PAR_CALLOC(float, npoints * 3);
    float* dpoint = points;
    float const* spoint = mesh->points;
    for (int t = 0; t < mesh->ntriangles; t++, spoint += 9, dpoint += 3) {
        float const* a = spoint;
        float const* b = spoint + 3;
        float const* c = spoint + 6;
        float const* p0 = dpoint;
        float const* p1 = dpoint + 3;
        float const* p2 = dpoint + 6;
        par_shapes__mix3(dpoint, a, b, 0.5);
        par_shapes__mix3(dpoint += 3, b, c, 0.5);
        par_shapes__mix3(dpoint += 3, a, c, 0.5);
        par_shapes__add3(dpoint += 3, a);
        par_shapes__add3(dpoint += 3, p0);
        par_shapes__add3(dpoint += 3, p2);
        par_shapes__add3(dpoint += 3, p0);
        par_shapes__add3(dpoint += 3, b);
        par_shapes__add3(dpoint += 3, p1);
        par_shapes__add3(dpoint += 3, p2);
        par_shapes__add3(dpoint += 3, p1);
        par_shapes__add3(dpoint += 3, c);
    }
    PAR_FREE(mesh->points);
    mesh->points = points;
    mesh->npoints = npoints;
    mesh->ntriangles = ntriangles;
}

par_shapes_mesh* par_shapes_create_subdivided_sphere(int nsubd)
{
    par_shapes_mesh* mesh = par_shapes_create_icosahedron();
    par_shapes_unweld(mesh, false);
    PAR_FREE(mesh->triangles);
    mesh->triangles = 0;
    while (nsubd--) {
        par_shapes__subdivide(mesh);
    }
    for (int i = 0; i < mesh->npoints; i++) {
        par_shapes__normalize3(mesh->points + i * 3);
    }
    mesh->triangles = PAR_MALLOC(PAR_SHAPES_T, 3 * mesh->ntriangles);
    for (int i = 0; i < mesh->ntriangles * 3; i++) {
        mesh->triangles[i] = i;
    }
    par_shapes_mesh* tmp = mesh;
    mesh = par_shapes_weld(mesh, 0.01, 0);
    par_shapes_free_mesh(tmp);
    par_shapes_compute_normals(mesh);
    return mesh;
}

par_shapes_mesh* par_shapes_create_rock(int seed, int subd)
{
    par_shapes_mesh* mesh = par_shapes_create_subdivided_sphere(subd);
    struct osn_context* ctx;
    par__simplex_noise(seed, &ctx);
    for (int p = 0; p < mesh->npoints; p++) {
        float* pt = mesh->points + p * 3;
        float a = 0.25, f = 1.0;
        double n = a * par__simplex_noise2(ctx, f * pt[0], f * pt[2]);
        a *= 0.5; f *= 2;
        n += a * par__simplex_noise2(ctx, f * pt[0], f * pt[2]);
        pt[0] *= 1 + 2 * n;
        pt[1] *= 1 + n;
        pt[2] *= 1 + 2 * n;
        if (pt[1] < 0) {
            pt[1] = -pow(-pt[1], 0.5) / 2;
        }
    }
    par__simplex_noise_free(ctx);
    par_shapes_compute_normals(mesh);
    return mesh;
}

par_shapes_mesh* par_shapes_clone(par_shapes_mesh const* mesh,
    par_shapes_mesh* clone)
{
    if (!clone) {
        clone = PAR_CALLOC(par_shapes_mesh, 1);
    }
    clone->npoints = mesh->npoints;
    clone->points = PAR_REALLOC(float, clone->points, 3 * clone->npoints);
    memcpy(clone->points, mesh->points, sizeof(float) * 3 * clone->npoints);
    clone->ntriangles = mesh->ntriangles;
    clone->triangles = PAR_REALLOC(PAR_SHAPES_T, clone->triangles, 3 *
        clone->ntriangles);
    memcpy(clone->triangles, mesh->triangles,
        sizeof(PAR_SHAPES_T) * 3 * clone->ntriangles);
    if (mesh->normals) {
        clone->normals = PAR_REALLOC(float, clone->normals, 3 * clone->npoints);
        memcpy(clone->normals, mesh->normals,
            sizeof(float) * 3 * clone->npoints);
    }
    if (mesh->tcoords) {
        clone->tcoords = PAR_REALLOC(float, clone->tcoords, 2 * clone->npoints);
        memcpy(clone->tcoords, mesh->tcoords,
            sizeof(float) * 2 * clone->npoints);
    }
    return clone;
}

static struct {
    float const* points;
    int gridsize;
} par_shapes__sort_context;

static int par_shapes__cmp1(const void *arg0, const void *arg1)
{
    const int g = par_shapes__sort_context.gridsize;

    // Convert arg0 into a flattened grid index.
    PAR_SHAPES_T d0 = *(const PAR_SHAPES_T*) arg0;
    float const* p0 = par_shapes__sort_context.points + d0 * 3;
    int i0 = (int) p0[0];
    int j0 = (int) p0[1];
    int k0 = (int) p0[2];
    int index0 = i0 + g * j0 + g * g * k0;

    // Convert arg1 into a flattened grid index.
    PAR_SHAPES_T d1 = *(const PAR_SHAPES_T*) arg1;
    float const* p1 = par_shapes__sort_context.points + d1 * 3;
    int i1 = (int) p1[0];
    int j1 = (int) p1[1];
    int k1 = (int) p1[2];
    int index1 = i1 + g * j1 + g * g * k1;

    // Return the ordering.
    if (index0 < index1) return -1;
    if (index0 > index1) return 1;
    return 0;
}

static void par_shapes__sort_points(par_shapes_mesh* mesh, int gridsize,
    PAR_SHAPES_T* sortmap)
{
    // Run qsort over a list of consecutive integers that get deferenced
    // within the comparator function; this creates a reorder mapping.
    for (int i = 0; i < mesh->npoints; i++) {
        sortmap[i] = i;
    }
    par_shapes__sort_context.gridsize = gridsize;
    par_shapes__sort_context.points = mesh->points;
    qsort(sortmap, mesh->npoints, sizeof(PAR_SHAPES_T), par_shapes__cmp1);

    // Apply the reorder mapping to the XYZ coordinate data.
    float* newpts = PAR_MALLOC(float, mesh->npoints * 3);
    PAR_SHAPES_T* invmap = PAR_MALLOC(PAR_SHAPES_T, mesh->npoints);
    float* dstpt = newpts;
    for (int i = 0; i < mesh->npoints; i++) {
        invmap[sortmap[i]] = i;
        float const* srcpt = mesh->points + 3 * sortmap[i];
        *dstpt++ = *srcpt++;
        *dstpt++ = *srcpt++;
        *dstpt++ = *srcpt++;
    }
    PAR_FREE(mesh->points);
    mesh->points = newpts;

    // Apply the inverse reorder mapping to the triangle indices.
    PAR_SHAPES_T* newinds = PAR_MALLOC(PAR_SHAPES_T, mesh->ntriangles * 3);
    PAR_SHAPES_T* dstind = newinds;
    PAR_SHAPES_T const* srcind = mesh->triangles;
    for (int i = 0; i < mesh->ntriangles * 3; i++) {
        *dstind++ = invmap[*srcind++];
    }
    PAR_FREE(mesh->triangles);
    mesh->triangles = newinds;

    // Cleanup.
    memcpy(sortmap, invmap, sizeof(PAR_SHAPES_T) * mesh->npoints);
    PAR_FREE(invmap);
}

static void par_shapes__weld_points(par_shapes_mesh* mesh, int gridsize,
    float epsilon, PAR_SHAPES_T* weldmap)
{
    // Each bin contains a "pointer" (really an index) to its first point.
    // We add 1 because 0 is reserved to mean that the bin is empty.
    // Since the points are spatially sorted, there's no need to store
    // a point count in each bin.
    PAR_SHAPES_T* bins = PAR_CALLOC(PAR_SHAPES_T,
        gridsize * gridsize * gridsize);
    int prev_binindex = -1;
    for (int p = 0; p < mesh->npoints; p++) {
        float const* pt = mesh->points + p * 3;
        int i = (int) pt[0];
        int j = (int) pt[1];
        int k = (int) pt[2];
        int this_binindex = i + gridsize * j + gridsize * gridsize * k;
        if (this_binindex != prev_binindex) {
            bins[this_binindex] = 1 + p;
        }
        prev_binindex = this_binindex;
    }

    // Examine all bins that intersect the epsilon-sized cube centered at each
    // point, and check for colocated points within those bins.
    float const* pt = mesh->points;
    int nremoved = 0;
    for (int p = 0; p < mesh->npoints; p++, pt += 3) {

        // Skip if this point has already been welded.
        if (weldmap[p] != p) {
            continue;
        }

        // Build a list of bins that intersect the epsilon-sized cube.
        int nearby[8];
        int nbins = 0;
        int minp[3], maxp[3];
        for (int c = 0; c < 3; c++) {
            minp[c] = (int) (pt[c] - epsilon);
            maxp[c] = (int) (pt[c] + epsilon);
        }
        for (int i = minp[0]; i <= maxp[0]; i++) {
            for (int j = minp[1]; j <= maxp[1]; j++) {
                for (int k = minp[2]; k <= maxp[2]; k++) {
                    int binindex = i + gridsize * j + gridsize * gridsize * k;
                    PAR_SHAPES_T binvalue = *(bins + binindex);
                    if (binvalue > 0) {
                        if (nbins == 8) {
                            printf("Epsilon value is too large.\n");
                            break;
                        }
                        nearby[nbins++] = binindex;
                    }
                }
            }
        }

        // Check for colocated points in each nearby bin.
        for (int b = 0; b < nbins; b++) {
            int binindex = nearby[b];
            PAR_SHAPES_T binvalue = bins[binindex];
            PAR_SHAPES_T nindex = binvalue - 1;
            assert(nindex < mesh->npoints);
            while (true) {

                // If this isn't "self" and it's colocated, then weld it!
                if (nindex != p && weldmap[nindex] == nindex) {
                    float const* thatpt = mesh->points + nindex * 3;
                    float dist2 = par_shapes__sqrdist3(thatpt, pt);
                    if (dist2 < epsilon) {
                        weldmap[nindex] = p;
                        nremoved++;
                    }
                }

                // Advance to the next point if possible.
                if (++nindex >= mesh->npoints) {
                    break;
                }

                // If the next point is outside the bin, then we're done.
                float const* nextpt = mesh->points + nindex * 3;
                int i = (int) nextpt[0];
                int j = (int) nextpt[1];
                int k = (int) nextpt[2];
                int nextbinindex = i + gridsize * j + gridsize * gridsize * k;
                if (nextbinindex != binindex) {
                    break;
                }
            }
        }
    }
    PAR_FREE(bins);

    // Apply the weldmap to the vertices.
    int npoints = mesh->npoints - nremoved;
    float* newpts = PAR_MALLOC(float, 3 * npoints);
    float* dst = newpts;
    PAR_SHAPES_T* condensed_map = PAR_MALLOC(PAR_SHAPES_T, mesh->npoints);
    PAR_SHAPES_T* cmap = condensed_map;
    float const* src = mesh->points;
    int ci = 0;
    for (int p = 0; p < mesh->npoints; p++, src += 3) {
        if (weldmap[p] == p) {
            *dst++ = src[0];
            *dst++ = src[1];
            *dst++ = src[2];
            *cmap++ = ci++;
        } else {
            *cmap++ = condensed_map[weldmap[p]];
        }
    }
    assert(ci == npoints);
    PAR_FREE(mesh->points);
    memcpy(weldmap, condensed_map, mesh->npoints * sizeof(PAR_SHAPES_T));
    PAR_FREE(condensed_map);
    mesh->points = newpts;
    mesh->npoints = npoints;

    // Apply the weldmap to the triangle indices and skip the degenerates.
    PAR_SHAPES_T const* tsrc = mesh->triangles;
    PAR_SHAPES_T* tdst = mesh->triangles;
    int ntriangles = 0;
    for (int i = 0; i < mesh->ntriangles; i++, tsrc += 3) {
        PAR_SHAPES_T a = weldmap[tsrc[0]];
        PAR_SHAPES_T b = weldmap[tsrc[1]];
        PAR_SHAPES_T c = weldmap[tsrc[2]];
        if (a != b && a != c && b != c) {
            assert(a < mesh->npoints);
            assert(b < mesh->npoints);
            assert(c < mesh->npoints);
            *tdst++ = a;
            *tdst++ = b;
            *tdst++ = c;
            ntriangles++;
        }
    }
    mesh->ntriangles = ntriangles;
}

par_shapes_mesh* par_shapes_weld(par_shapes_mesh const* mesh, float epsilon,
    PAR_SHAPES_T* weldmap)
{
    par_shapes_mesh* clone = par_shapes_clone(mesh, 0);
    float aabb[6];
    int gridsize = 20;
    float maxcell = gridsize - 1;
    par_shapes_compute_aabb(clone, aabb);
    float scale[3] = {
        aabb[3] == aabb[0] ? 1.0f : maxcell / (aabb[3] - aabb[0]),
        aabb[4] == aabb[1] ? 1.0f : maxcell / (aabb[4] - aabb[1]),
        aabb[5] == aabb[2] ? 1.0f : maxcell / (aabb[5] - aabb[2]),
    };
    par_shapes_translate(clone, -aabb[0], -aabb[1], -aabb[2]);
    par_shapes_scale(clone, scale[0], scale[1], scale[2]);
    PAR_SHAPES_T* sortmap = PAR_MALLOC(PAR_SHAPES_T, mesh->npoints);
    par_shapes__sort_points(clone, gridsize, sortmap);
    bool owner = false;
    if (!weldmap) {
        owner = true;
        weldmap = PAR_MALLOC(PAR_SHAPES_T, mesh->npoints);
    }
    for (int i = 0; i < mesh->npoints; i++) {
        weldmap[i] = i;
    }
    par_shapes__weld_points(clone, gridsize, epsilon, weldmap);
    if (owner) {
        PAR_FREE(weldmap);
    } else {
        PAR_SHAPES_T* newmap = PAR_MALLOC(PAR_SHAPES_T, mesh->npoints);
        for (int i = 0; i < mesh->npoints; i++) {
            newmap[i] = weldmap[sortmap[i]];
        }
        memcpy(weldmap, newmap, sizeof(PAR_SHAPES_T) * mesh->npoints);
        PAR_FREE(newmap);
    }
    PAR_FREE(sortmap);
    par_shapes_scale(clone, 1.0 / scale[0], 1.0 / scale[1], 1.0 / scale[2]);
    par_shapes_translate(clone, aabb[0], aabb[1], aabb[2]);
    return clone;
}

// -----------------------------------------------------------------------------
// BEGIN OPEN SIMPLEX NOISE
// -----------------------------------------------------------------------------

#define STRETCH_CONSTANT_2D (-0.211324865405187)  // (1 / sqrt(2 + 1) - 1 ) / 2;
#define SQUISH_CONSTANT_2D (0.366025403784439)  // (sqrt(2 + 1) -1) / 2;
#define STRETCH_CONSTANT_3D (-1.0 / 6.0)  // (1 / sqrt(3 + 1) - 1) / 3;
#define SQUISH_CONSTANT_3D (1.0 / 3.0)  // (sqrt(3+1)-1)/3;
#define STRETCH_CONSTANT_4D (-0.138196601125011)  // (1 / sqrt(4 + 1) - 1) / 4;
#define SQUISH_CONSTANT_4D (0.309016994374947)  // (sqrt(4 + 1) - 1) / 4;

#define NORM_CONSTANT_2D (47.0)
#define NORM_CONSTANT_3D (103.0)
#define NORM_CONSTANT_4D (30.0)

#define DEFAULT_SEED (0LL)

struct osn_context {
    int16_t* perm;
    int16_t* permGradIndex3D;
};

#define ARRAYSIZE(x) (sizeof((x)) / sizeof((x)[0]))

/*
 * Gradients for 2D. They approximate the directions to the
 * vertices of an octagon from the center.
 */
static const int8_t gradients2D[] = {
    5, 2, 2, 5, -5, 2, -2, 5, 5, -2, 2, -5, -5, -2, -2, -5,
};

/*
 * Gradients for 3D. They approximate the directions to the
 * vertices of a rhombicuboctahedron from the center, skewed so
 * that the triangular and square facets can be inscribed inside
 * circles of the same radius.
 */
static const signed char gradients3D[] = {
    -11, 4, 4, -4, 11, 4, -4, 4, 11, 11, 4, 4, 4, 11, 4, 4, 4, 11, -11, -4, 4,
    -4, -11, 4, -4, -4, 11, 11, -4, 4, 4, -11, 4, 4, -4, 11, -11, 4, -4, -4, 11,
    -4, -4, 4, -11, 11, 4, -4, 4, 11, -4, 4, 4, -11, -11, -4, -4, -4, -11, -4,
    -4, -4, -11, 11, -4, -4, 4, -11, -4, 4, -4, -11,
};

/*
 * Gradients for 4D. They approximate the directions to the
 * vertices of a disprismatotesseractihexadecachoron from the center,
 * skewed so that the tetrahedral and cubic facets can be inscribed inside
 * spheres of the same radius.
 */
static const signed char gradients4D[] = {
    3, 1, 1, 1, 1, 3, 1, 1, 1, 1, 3, 1, 1, 1, 1, 3, -3, 1, 1, 1, -1, 3, 1, 1,
    -1, 1, 3, 1, -1, 1, 1, 3, 3, -1, 1, 1, 1, -3, 1, 1, 1, -1, 3, 1, 1, -1, 1,
    3, -3, -1, 1, 1, -1, -3, 1, 1, -1, -1, 3, 1, -1, -1, 1, 3, 3, 1, -1, 1, 1,
    3, -1, 1, 1, 1, -3, 1, 1, 1, -1, 3, -3, 1, -1, 1, -1, 3, -1, 1, -1, 1, -3,
    1, -1, 1, -1, 3, 3, -1, -1, 1, 1, -3, -1, 1, 1, -1, -3, 1, 1, -1, -1, 3, -3,
    -1, -1, 1, -1, -3, -1, 1, -1, -1, -3, 1, -1, -1, -1, 3, 3, 1, 1, -1, 1, 3,
    1, -1, 1, 1, 3, -1, 1, 1, 1, -3, -3, 1, 1, -1, -1, 3, 1, -1, -1, 1, 3, -1,
    -1, 1, 1, -3, 3, -1, 1, -1, 1, -3, 1, -1, 1, -1, 3, -1, 1, -1, 1, -3, -3,
    -1, 1, -1, -1, -3, 1, -1, -1, -1, 3, -1, -1, -1, 1, -3, 3, 1, -1, -1, 1, 3,
    -1, -1, 1, 1, -3, -1, 1, 1, -1, -3, -3, 1, -1, -1, -1, 3, -1, -1, -1, 1, -3,
    -1, -1, 1, -1, -3, 3, -1, -1, -1, 1, -3, -1, -1, 1, -1, -3, -1, 1, -1, -1,
    -3, -3, -1, -1, -1, -1, -3, -1, -1, -1, -1, -3, -1, -1, -1, -1, -3,
};

static double extrapolate2(
    struct osn_context* ctx, int xsb, int ysb, double dx, double dy)
{
    int16_t* perm = ctx->perm;
    int index = perm[(perm[xsb & 0xFF] + ysb) & 0xFF] & 0x0E;
    return gradients2D[index] * dx + gradients2D[index + 1] * dy;
}

static inline int fastFloor(double x)
{
    int xi = (int) x;
    return x < xi ? xi - 1 : xi;
}

static int allocate_perm(struct osn_context* ctx, int nperm, int ngrad)
{
    PAR_FREE(ctx->perm);
    PAR_FREE(ctx->permGradIndex3D);
    ctx->perm = PAR_MALLOC(int16_t, nperm);
    if (!ctx->perm) {
        return -ENOMEM;
    }
    ctx->permGradIndex3D = PAR_MALLOC(int16_t, ngrad);
    if (!ctx->permGradIndex3D) {
        PAR_FREE(ctx->perm);
        return -ENOMEM;
    }
    return 0;
}

static int par__simplex_noise(int64_t seed, struct osn_context** ctx)
{
    int rc;
    int16_t source[256];
    int i;
    int16_t* perm;
    int16_t* permGradIndex3D;
    *ctx = PAR_MALLOC(struct osn_context, 1);
    if (!(*ctx)) {
        return -ENOMEM;
    }
    (*ctx)->perm = NULL;
    (*ctx)->permGradIndex3D = NULL;
    rc = allocate_perm(*ctx, 256, 256);
    if (rc) {
        PAR_FREE(*ctx);
        return rc;
    }
    perm = (*ctx)->perm;
    permGradIndex3D = (*ctx)->permGradIndex3D;
    for (i = 0; i < 256; i++) {
        source[i] = (int16_t) i;
    }
    seed = seed * 6364136223846793005LL + 1442695040888963407LL;
    seed = seed * 6364136223846793005LL + 1442695040888963407LL;
    seed = seed * 6364136223846793005LL + 1442695040888963407LL;
    for (i = 255; i >= 0; i--) {
        seed = seed * 6364136223846793005LL + 1442695040888963407LL;
        int r = (int) ((seed + 31) % (i + 1));
        if (r < 0)
            r += (i + 1);
        perm[i] = source[r];
        permGradIndex3D[i] =
            (short) ((perm[i] % (ARRAYSIZE(gradients3D) / 3)) * 3);
        source[r] = source[i];
    }
    return 0;
}

static void par__simplex_noise_free(struct osn_context* ctx)
{
    if (!ctx)
        return;
    if (ctx->perm) {
        PAR_FREE(ctx->perm);
        ctx->perm = NULL;
    }
    if (ctx->permGradIndex3D) {
        PAR_FREE(ctx->permGradIndex3D);
        ctx->permGradIndex3D = NULL;
    }
    PAR_FREE(ctx);
}

static double par__simplex_noise2(struct osn_context* ctx, double x, double y)
{
    // Place input coordinates onto grid.
    double stretchOffset = (x + y) * STRETCH_CONSTANT_2D;
    double xs = x + stretchOffset;
    double ys = y + stretchOffset;

    // Floor to get grid coordinates of rhombus (stretched square) super-cell
    // origin.
    int xsb = fastFloor(xs);
    int ysb = fastFloor(ys);

    // Skew out to get actual coordinates of rhombus origin. We'll need these
    // later.
    double squishOffset = (xsb + ysb) * SQUISH_CONSTANT_2D;
    double xb = xsb + squishOffset;
    double yb = ysb + squishOffset;

    // Compute grid coordinates relative to rhombus origin.
    double xins = xs - xsb;
    double yins = ys - ysb;

    // Sum those together to get a value that determines which region we're in.
    double inSum = xins + yins;

    // Positions relative to origin point.
    double dx0 = x - xb;
    double dy0 = y - yb;

    // We'll be defining these inside the next block and using them afterwards.
    double dx_ext, dy_ext;
    int xsv_ext, ysv_ext;

    double value = 0;

    // Contribution (1,0)
    double dx1 = dx0 - 1 - SQUISH_CONSTANT_2D;
    double dy1 = dy0 - 0 - SQUISH_CONSTANT_2D;
    double attn1 = 2 - dx1 * dx1 - dy1 * dy1;
    if (attn1 > 0) {
        attn1 *= attn1;
        value += attn1 * attn1 * extrapolate2(ctx, xsb + 1, ysb + 0, dx1, dy1);
    }

    // Contribution (0,1)
    double dx2 = dx0 - 0 - SQUISH_CONSTANT_2D;
    double dy2 = dy0 - 1 - SQUISH_CONSTANT_2D;
    double attn2 = 2 - dx2 * dx2 - dy2 * dy2;
    if (attn2 > 0) {
        attn2 *= attn2;
        value += attn2 * attn2 * extrapolate2(ctx, xsb + 0, ysb + 1, dx2, dy2);
    }

    if (inSum <= 1) {  // We're inside the triangle (2-Simplex) at (0,0)
        double zins = 1 - inSum;
        if (zins > xins || zins > yins) {
            if (xins > yins) {
                xsv_ext = xsb + 1;
                ysv_ext = ysb - 1;
                dx_ext = dx0 - 1;
                dy_ext = dy0 + 1;
            } else {
                xsv_ext = xsb - 1;
                ysv_ext = ysb + 1;
                dx_ext = dx0 + 1;
                dy_ext = dy0 - 1;
            }
        } else {  //(1,0) and (0,1) are the closest two vertices.
            xsv_ext = xsb + 1;
            ysv_ext = ysb + 1;
            dx_ext = dx0 - 1 - 2 * SQUISH_CONSTANT_2D;
            dy_ext = dy0 - 1 - 2 * SQUISH_CONSTANT_2D;
        }
    } else {  // We're inside the triangle (2-Simplex) at (1,1)
        double zins = 2 - inSum;
        if (zins < xins || zins < yins) {
            if (xins > yins) {
                xsv_ext = xsb + 2;
                ysv_ext = ysb + 0;
                dx_ext = dx0 - 2 - 2 * SQUISH_CONSTANT_2D;
                dy_ext = dy0 + 0 - 2 * SQUISH_CONSTANT_2D;
            } else {
                xsv_ext = xsb + 0;
                ysv_ext = ysb + 2;
                dx_ext = dx0 + 0 - 2 * SQUISH_CONSTANT_2D;
                dy_ext = dy0 - 2 - 2 * SQUISH_CONSTANT_2D;
            }
        } else {  //(1,0) and (0,1) are the closest two vertices.
            dx_ext = dx0;
            dy_ext = dy0;
            xsv_ext = xsb;
            ysv_ext = ysb;
        }
        xsb += 1;
        ysb += 1;
        dx0 = dx0 - 1 - 2 * SQUISH_CONSTANT_2D;
        dy0 = dy0 - 1 - 2 * SQUISH_CONSTANT_2D;
    }

    // Contribution (0,0) or (1,1)
    double attn0 = 2 - dx0 * dx0 - dy0 * dy0;
    if (attn0 > 0) {
        attn0 *= attn0;
        value += attn0 * attn0 * extrapolate2(ctx, xsb, ysb, dx0, dy0);
    }

    // Extra Vertex
    double attn_ext = 2 - dx_ext * dx_ext - dy_ext * dy_ext;
    if (attn_ext > 0) {
        attn_ext *= attn_ext;
        value += attn_ext * attn_ext *
            extrapolate2(ctx, xsv_ext, ysv_ext, dx_ext, dy_ext);
    }

    return value / NORM_CONSTANT_2D;
}

void par_shapes_remove_degenerate(par_shapes_mesh* mesh, float mintriarea)
{
    int ntriangles = 0;
    PAR_SHAPES_T* triangles = PAR_MALLOC(PAR_SHAPES_T, mesh->ntriangles * 3);
    PAR_SHAPES_T* dst = triangles;
    PAR_SHAPES_T const* src = mesh->triangles;
    float next[3], prev[3], cp[3];
    float mincplen2 = (mintriarea * 2) * (mintriarea * 2);
    for (int f = 0; f < mesh->ntriangles; f++, src += 3) {
        float const* pa = mesh->points + 3 * src[0];
        float const* pb = mesh->points + 3 * src[1];
        float const* pc = mesh->points + 3 * src[2];
        par_shapes__copy3(next, pb);
        par_shapes__subtract3(next, pa);
        par_shapes__copy3(prev, pc);
        par_shapes__subtract3(prev, pa);
        par_shapes__cross3(cp, next, prev);
        float cplen2 = par_shapes__dot3(cp, cp);
        if (cplen2 >= mincplen2) {
            *dst++ = src[0];
            *dst++ = src[1];
            *dst++ = src[2];
            ntriangles++;
        }
    }
    mesh->ntriangles = ntriangles;
    PAR_FREE(mesh->triangles);
    mesh->triangles = triangles;
}

#endif // PAR_SHAPES_IMPLEMENTATION
#endif // PAR_SHAPES_H
framebuffer* current_fb;
framebuffer* default_fb;
Image* active_texture;
draw_mode cmode;
vertex::uv_t current_uv;
vertex::color_t current_color;
std::vector<vertex> current_buffer;
std::stack<Matrix4<float>> matrix_stack;
Vector4<float> texture2D(const Image& img, const Vector2<float>& uv){
    const unsigned char* cptr = (const unsigned char*)(img.data);
    Vector4<float> ret;
    int x = std::min((unsigned int)(uv.x * (img.width)), img.width - 1);
    int y = std::min((unsigned int)(uv.y * (img.height)), img.height - 1);
    
    ret.x = cptr[(y * img.width + x) * 4 + 0] / 255.0f;
    ret.y = cptr[(y * img.width + x) * 4 + 1] / 255.0f;
    ret.z = cptr[(y * img.width + x) * 4 + 2] / 255.0f;
    ret.w = cptr[(y * img.width + x) * 4 + 3] / 255.0f;

    return ret;
}
void ClearBackground(Color colu){
    Vector3<float> col(colu.cast<float>().head3() * (1.0f / 255.0f));
    std::fill(current_fb->depth_buffer.data.get(), current_fb->depth_buffer.data.get() + current_fb->resolution.x * current_fb->resolution.y, framebuffer::empty_depth);
    std::fill(current_fb->color_buffer.data.get(), current_fb->color_buffer.data.get() + current_fb->resolution.x * current_fb->resolution.y, col);            
}
Image GenImageChecked(int width, int height, int checksX, int checksY, Color col1, Color col2){
    std::unique_ptr<Color[]> pixels = std::make_unique<Color[]>(width * height);

    for (int y = 0; y < height; y++){
        for (int x = 0; x < width; x++){
            if ((x / checksX + y / checksY) % 2 == 0) pixels[y * width + x] = col1;
            else pixels[y * width + x] = col2;
        }
    }
    return Image(width, height, std::move(pixels));
}
void depthblend_framebuffers(framebuffer& target, const framebuffer& op){
    assert(target.resolution.x == op.resolution.x);
    assert(target.resolution.y == op.resolution.y);
    const size_t bound = op.resolution.x * op.resolution.y;
    for(size_t i = 0; i < bound;i++){
        framebuffer::color_t::scalar op_blendsover = op.depth_buffer.data[i] < target.depth_buffer.data[i];
        framebuffer::color_t::scalar one_minus_blendsover = framebuffer::color_t::scalar(1) - op_blendsover;
        target.color_buffer.data[i] = op_blendsover * op.color_buffer.data[i] + target.color_buffer.data[i] * one_minus_blendsover;
        target.depth_buffer.data[i] = std::min(op.depth_buffer.data[i], target.depth_buffer.data[i]);
    }
}
template<typename T>
bool is_in_clip_cube(const Vector3<T>& x){
    return x.x >= T(-1) &&
           x.x <= T( 1) &&
           x.y >= T(-1) &&
           x.y <= T( 1) &&
           x.z >= T(-1) &&
           x.z <= T( 1);
}
Mesh GenMeshSphere(float radius, int rings, int slices){
    Mesh mesh;

    if ((rings >= 3) && (slices >= 3)){
        par_shapes_mesh *sphere = par_shapes_create_parametric_sphere(slices, rings);
        par_shapes_scale(sphere, radius, radius, radius);
        // NOTE: Soft normals are computed internally

        mesh.vertices  = (float*)std::malloc(sphere->ntriangles * 3 * 3 * sizeof(float));
        mesh.texcoords = (float*)std::malloc(sphere->ntriangles * 3 * 2 * sizeof(float));
        //mesh.normals =  (float *) std::malloc(sphere->ntriangles*3*3*sizeof(float));

        mesh.vertexCount = sphere->ntriangles*3;
        mesh.triangleCount = sphere->ntriangles;

        for (int k = 0; k < mesh.vertexCount; k++){
            mesh.vertices[k*3] = sphere->points[sphere->triangles[k] * 3];
            mesh.vertices[k*3 + 1] = sphere->points[sphere->triangles[k] * 3 + 1];
            mesh.vertices[k*3 + 2] = sphere->points[sphere->triangles[k] * 3 + 2];

            //mesh.normals[k*3] = sphere->normals[sphere->triangles[k]*3];
            //mesh.normals[k*3 + 1] = sphere->normals[sphere->triangles[k]*3 + 1];
            //mesh.normals[k*3 + 2] = sphere->normals[sphere->triangles[k]*3 + 2];

            mesh.texcoords[k*2] = sphere->tcoords[sphere->triangles[k]*2];
            mesh.texcoords[k*2 + 1] = sphere->tcoords[sphere->triangles[k]*2 + 1];
        }

        par_shapes_free_mesh(sphere);
    }

    return mesh;
}
template<bool textured>
void draw_triangle_already_projected(framebuffer& img, vertex p1, vertex p2, vertex p3, const Image* texture){
    Vector4<float> clipp1 = one_extend(p1.pos);
    Vector4<float> clipp2 = one_extend(p2.pos);
    Vector4<float> clipp3 = one_extend(p3.pos);

    Vector2<int> p1_screen = img.clip2screen(Vector2<float>{clipp1.x, clipp1.y});
    Vector2<int> p2_screen = img.clip2screen(Vector2<float>{clipp2.x, clipp2.y});
    Vector2<int> p3_screen = img.clip2screen(Vector2<float>{clipp3.x, clipp3.y});

    if(p1_screen.y > p2_screen.y){
        std::swap(p1_screen, p2_screen);
        std::swap(clipp1, clipp2);
        std::swap(p1, p2);
    }
    if(p2_screen.y > p3_screen.y){
        std::swap(p2_screen, p3_screen);
        std::swap(clipp2, clipp3);
        std::swap(p2, p3);
    }
    if(p1_screen.y > p2_screen.y){
        std::swap(p1_screen, p2_screen);
        std::swap(clipp1, clipp2);
        std::swap(p1, p2);
    }
    using Vector2i = Vector2<int>;
    using Vector2f = Vector2<float>;
    using Vector3f = Vector3<float>;

    Vector2i mine = p1_screen.cwiseMin(p2_screen.cwiseMin(p3_screen));//.cwiseMax(Vector2i{0,0}); 
    Vector2i maxe = (p1_screen.cwiseMax(p2_screen.cwiseMax(p3_screen))).cwiseMin(img.resolution.cast<int>());
    barycentric_triangle_function<float> bary(clipp1, clipp2, clipp3);
    for (int y = mine.y; y <= maxe.y; y++) {
        int x1, x2;
        
        if (y >= p1_screen.y && y <= p2_screen.y) {
            if(p2_screen.y == p1_screen.y){
                x1 = (std::min(p1_screen.x, p2_screen.x));
                x2 = (std::max(p1_screen.x, p2_screen.x));
            }
            else{
                float t12 = static_cast<float>(y - p1_screen.y) / (p2_screen.y - p1_screen.y);
                float t13 = static_cast<float>(y - p1_screen.y) / (p3_screen.y - p1_screen.y);
                x1 = p1_screen.x + t12 * (p2_screen.x - p1_screen.x);
                x2 = p1_screen.x + t13 * (p3_screen.x - p1_screen.x);
            }
        }
        else{
            if(p3_screen.y == p1_screen.y){
                x1 = std::min(std::min(p1_screen.x, p2_screen.x), p3_screen.x);
                x2 = std::max(std::max(p1_screen.x, p2_screen.x), p3_screen.x);
            }
            else{
                float t13 = static_cast<float>(y - p1_screen.y) / (p3_screen.y - p1_screen.y);
                float t23 = static_cast<float>(y - p2_screen.y) / (p3_screen.y - p2_screen.y);
                x1 = p1_screen.x + t13 * (p3_screen.x - p1_screen.x);
                x2 = p2_screen.x + t23 * (p3_screen.x - p2_screen.x);
            }
        }
        
        if (x1 > x2) {
            std::swap(x1, x2);
        }
        x1 -= 1;
        x2 += 1;
        //std::cout << "Y: " << y << " , Raschtering from " << x1 << " to " << x2 << "\n";
        for (int x = x1; x <= x2; ++x) {
            Vector2<float> clip = img.screen2clip(Vector2i{x, y});
            Vector3<float> linear = bary.linear(clip);
            if(linear.maxCoeff() <= 1.0f && linear.minCoeff() >= 0.0f){
                Vector3f one_over_ws = linear * bary.one_over_ws;
                float isum = 1.0f / (one_over_ws.x + one_over_ws.y + one_over_ws.z);
                Vector4<float> frag_color = zero_extend(bary.perspective_correct2(linear, one_over_ws, isum, clip, p1.color, p2.color, p3.color));
                if constexpr(textured){
                    Vector2f beval = bary.perspective_correct2(linear, one_over_ws, isum, clip, p1.uv, p2.uv, p3.uv);
                    frag_color *= texture2D(*texture, beval);
                }
                float zeval = bary.perspective_correct2(linear, one_over_ws, isum, clip, clipp1.z, clipp2.z, clipp3.z);
                //std::cout << beval.transpose() << "\n";
                img.paint_pixeli(x, y, Vector3<float>{frag_color.x,frag_color.y,frag_color.z}, 1.0f, zeval);
            }
        }
    }
}
template<bool textured>
void draw_triangle(framebuffer& img, const Matrix4<float>& mat, vertex p1, vertex p2, vertex p3, const Image* texture){
    //Matrix4<float> mat = cam.matrix(img.resolution.x, img.resolution.y);
    Vector4<float> clipp1 = one_extend(p1.pos);clipp1 = (mat * clipp1).homogenize();
    Vector4<float> clipp2 = one_extend(p2.pos);clipp2 = (mat * clipp2).homogenize();
    Vector4<float> clipp3 = one_extend(p3.pos);clipp3 = (mat * clipp3).homogenize();
    if(!(is_in_clip_cube(clipp1.head3()) || is_in_clip_cube(clipp2.head3()) || is_in_clip_cube(clipp3.head3()))){
        return;
    }
    //std::cout << clipp1.transpose() << "\n\n";
    Vector2<int> p1_screen = img.clip2screen(Vector2<float>{clipp1.x, clipp1.y});
    Vector2<int> p2_screen = img.clip2screen(Vector2<float>{clipp2.x, clipp2.y});
    Vector2<int> p3_screen = img.clip2screen(Vector2<float>{clipp3.x, clipp3.y});

    if(p1_screen.y > p2_screen.y){
        std::swap(p1_screen, p2_screen);
        std::swap(clipp1, clipp2);
        std::swap(p1, p2);
    }
    if(p2_screen.y > p3_screen.y){
        std::swap(p2_screen, p3_screen);
        std::swap(clipp2, clipp3);
        std::swap(p2, p3);
    }
    if(p1_screen.y > p2_screen.y){
        std::swap(p1_screen, p2_screen);
        std::swap(clipp1, clipp2);
        std::swap(p1, p2);
    }
    using Vector2i = Vector2<int>;
    using Vector2f = Vector2<float>;
    using Vector3f = Vector3<float>;

    Vector2i mine = p1_screen.cwiseMin(p2_screen.cwiseMin(p3_screen));//.cwiseMax(Vector2i{0,0}); 
    Vector2i maxe = (p1_screen.cwiseMax(p2_screen.cwiseMax(p3_screen))).cwiseMin(img.resolution.cast<int>());
    barycentric_triangle_function<float> bary(clipp1, clipp2, clipp3);
    for (int y = mine.y; y <= maxe.y; y++) {
        int x1, x2;
        if (y >= p1_screen.y && y <= p2_screen.y) {
            if(p2_screen.y == p1_screen.y){
                x1 = (std::min(p1_screen.x, p2_screen.x));
                x2 = (std::max(p1_screen.x, p2_screen.x));
            }
            else{
                float t12 = static_cast<float>(y - p1_screen.y) / (p2_screen.y - p1_screen.y);
                float t13 = static_cast<float>(y - p1_screen.y) / (p3_screen.y - p1_screen.y);
                x1 = p1_screen.x + t12 * (p2_screen.x - p1_screen.x);
                x2 = p1_screen.x + t13 * (p3_screen.x - p1_screen.x);
            }
        }
        else{
            if(p3_screen.y == p1_screen.y){
                x1 = std::min(std::min(p1_screen.x, p2_screen.x), p3_screen.x);
                x2 = std::max(std::max(p1_screen.x, p2_screen.x), p3_screen.x);
            }
            else{
                float t13 = static_cast<float>(y - p1_screen.y) / (p3_screen.y - p1_screen.y);
                float t23 = static_cast<float>(y - p2_screen.y) / (p3_screen.y - p2_screen.y);
                x1 = p1_screen.x + t13 * (p3_screen.x - p1_screen.x);
                x2 = p2_screen.x + t23 * (p3_screen.x - p2_screen.x);
            }
        }
        
        if (x1 > x2) {
            std::swap(x1, x2);
        }
        x1 -= 1;
        x2 += 1;
        //std::cout << "Y: " << y << " , Raschtering from " << x1 << " to " << x2 << "\n";
        for (int x = x1; x <= x2; ++x) {
            Vector2<float> clip = img.screen2clip(Vector2i{x, y});
            Vector3<float> linear = bary.linear(clip);
            if(linear.maxCoeff() <= 1.0f && linear.minCoeff() >= 0.0f){
                Vector3f one_over_ws = linear * bary.one_over_ws;
                float isum = 1.0f / (one_over_ws.x + one_over_ws.y + one_over_ws.z);
                Vector4<float> frag_color = zero_extend(bary.perspective_correct2(linear, one_over_ws, isum, clip, p1.color, p2.color, p3.color));
                if constexpr(textured){
                    Vector2f beval = bary.perspective_correct2(linear, one_over_ws, isum, clip, p1.uv, p2.uv, p3.uv);
                    //Vector2f beval = p1.uv * linear.x + p2.uv * linear.y + p3.uv * linear.z;
                    frag_color *= texture2D(*texture, beval);
                }
                float zeval = bary.perspective_correct2(linear, one_over_ws, isum, clip, clipp1.z, clipp2.z, clipp3.z);
                //std::cout << beval.transpose() << "\n";
                if(zeval <= 1.0 && zeval >= -1.0)
                    img.paint_pixeli(x, y, Vector3<float>{frag_color.x,frag_color.y,frag_color.z}, 1.0f, zeval);
            }
        }
    }
}
void rlBegin(draw_mode mode){
    current_buffer.resize(1);
    cmode = mode;
}

void rlVertex3f(float x, float y, float z){
    current_buffer.push_back(vertex{.pos = Vector3<float>{x, y, z}, .uv = current_uv, .color = current_color});
}
void rlVertex2f(float x, float y){
    rlVertex3f(x, y, 0.0f);
}
void rlColor3f(float r, float g, float b){
    current_color = vertex::color_t{r, g, b};
    if(!current_buffer.empty()){
        current_buffer.back().color = vertex::color_t{r, g, b};
    }
}
void rlTexCoord2f(float r, float g){
    current_uv = vertex::uv_t{r, g};
    if(!current_buffer.empty()){
        current_buffer.back().uv = vertex::uv_t{r, g};
    }
}
void BeginTextureMode(framebuffer& fb){
    current_fb = &fb;
}
void EndTextureMode(){
    current_fb = default_fb;
}
void set_texture(Image* image){
    active_texture = image;
}
void unset_texture(){
    active_texture = nullptr;
}
void rlEnd(){
    if(cmode == triangles){
        while(current_buffer.size() >= 3){
            if(active_texture == nullptr)
                draw_triangle(*current_fb, matrix_stack.top(), current_buffer[current_buffer.size() - 3], current_buffer[current_buffer.size() - 2], current_buffer[current_buffer.size() - 1]);
            else
                draw_triangle<true>(*current_fb, matrix_stack.top(), current_buffer[current_buffer.size() - 3], current_buffer[current_buffer.size() - 2], current_buffer[current_buffer.size() - 1], active_texture);
            current_buffer.erase(current_buffer.end() - 3, current_buffer.end());
        }
    }
    cmode = nothing;
}
void DrawMesh(Mesh mesh, Matrix4<float> transform){
    rlBegin(triangles);
    for(int i = 0;i < mesh.vertexCount;i += 3){
        Vector3<float> v1{mesh.vertices[i * 3 + 0], mesh.vertices[i * 3 + 1], mesh.vertices[i * 3 + 2]};
        Vector3<float> v2{mesh.vertices[i * 3 + 3], mesh.vertices[i * 3 + 4], mesh.vertices[i * 3 + 5]};
        Vector3<float> v3{mesh.vertices[i * 3 + 6], mesh.vertices[i * 3 + 7], mesh.vertices[i * 3 + 8]};
        v1 = (transform * one_extend(v1)).head3();
        v2 = (transform * one_extend(v2)).head3();
        v3 = (transform * one_extend(v3)).head3();
        rlVertex3f(v1.x, v1.y, v1.z);
        rlColor3f(0.2,1,0.2);
        rlVertex3f(v2.x, v2.y, v2.z);
        rlColor3f(0.2,1,0.2);
        rlVertex3f(v3.x, v3.y, v3.z);
        rlColor3f(0.2,1,0.2);
    }
    rlEnd();
}
void DrawTriangleStrip(const Vector2<float> *points, int pointCount, Color color)
{
    if (pointCount >= 3)
    {
        rlBegin(triangles);
            rlColor3f(color.x / 255.0f, color.y / 255.0f, color.z / 255.0f);

            for (int i = 2; i < pointCount; i++)
            {
                if ((i%2) == 0)
                {
                    rlVertex2f(points[i].x, points[i].y);
                    rlVertex2f(points[i - 2].x, points[i - 2].y);
                    rlVertex2f(points[i - 1].x, points[i - 1].y);
                }
                else
                {
                    rlVertex2f(points[i].x, points[i].y);
                    rlVertex2f(points[i - 1].x, points[i - 1].y);
                    rlVertex2f(points[i - 2].x, points[i - 2].y);
                }
            }
        rlEnd();
    }
}
void DrawBillboardLineEx(Vector3<float> startPos, Vector3<float> endPos, float thick, Color color){
    using std::sqrt;
    using std::hypot;
    Vector4<float> sph = one_extend(startPos);
    Vector4<float> eph = one_extend(endPos);
    Matrix4<float> mat = matrix_stack.top();
    Vector4 sph_trf = (mat * sph).homogenize();
    Vector4 eph_trf = (mat * eph).homogenize();
    Vector2<float> delta = {eph_trf.x - sph_trf.x, eph_trf.y - sph_trf.y};

    float length = hypot(delta.x, delta.y);

    if ((length > 0) && (thick > 0))
    {
        float scale = thick/(2*length);
        Vector2<float> radius = { -scale*delta.y, scale*delta.x };
        Vector4<float> strip[4] = {
            { sph_trf.x - radius.x, sph_trf.y - radius.y , sph_trf.z, /*w needed?*/ 0.0f},
            { sph_trf.x + radius.x, sph_trf.y + radius.y , sph_trf.z, /*w needed?*/ 0.0f},
            { eph_trf.x - radius.x, eph_trf.y - radius.y , sph_trf.z, /*w needed?*/ 0.0f},
            { eph_trf.x + radius.x, eph_trf.y + radius.y , sph_trf.z, /*w needed?*/ 0.0f}
        };
        vertex v1{.pos = strip[0].head3(), .uv = Vector2<float>{0,0}, .color = Vector3<float>{color.x / 255.0f, color.y / 255.0f, color.z / 255.0f}};
        vertex v2{.pos = strip[1].head3(), .uv = Vector2<float>{0,0}, .color = Vector3<float>{color.x / 255.0f, color.y / 255.0f, color.z / 255.0f}};
        vertex v3{.pos = strip[2].head3(), .uv = Vector2<float>{0,0}, .color = Vector3<float>{color.x / 255.0f, color.y / 255.0f, color.z / 255.0f}};
        vertex v4{.pos = strip[3].head3(), .uv = Vector2<float>{0,0}, .color = Vector3<float>{color.x / 255.0f, color.y / 255.0f, color.z / 255.0f}};
        draw_triangle_already_projected<false>(*current_fb, v1, v2, v3);
        draw_triangle_already_projected<false>(*current_fb, v2, v3, v4);
        //DrawTriangleStrip(strip, 4, color);
    }
}
void DrawLineEx(Vector2<float> startPos, Vector2<float> endPos, float thick, Color color){
    using std::sqrt;
    using std::hypot;
    Vector2<float> delta = { endPos.x - startPos.x, endPos.y - startPos.y };
    float length = hypot(delta.x, delta.y);

    if ((length > 0) && (thick > 0))
    {
        float scale = thick/(2*length);
        Vector2<float> radius = { -scale*delta.y, scale*delta.x };
        Vector2<float> strip[4] = {
            { startPos.x - radius.x, startPos.y - radius.y },
            { startPos.x + radius.x, startPos.y + radius.y },
            { endPos.x - radius.x, endPos.y - radius.y },
            { endPos.x + radius.x, endPos.y + radius.y }
        };

        DrawTriangleStrip(strip, 4, color);
    }
}

void DrawRectangle(Vector2<float> pos, Vector2<float> ext){
    (void)pos;
    (void)ext;
}
void InitWindow(unsigned w, unsigned h){
    default_fb = new framebuffer(w, h);
    current_fb = default_fb;
    matrix_stack.push(ortho<float>(0, w, h, 0, -1, 1));
    active_texture = nullptr;
}
void outputPPM(const framebuffer& fb, const std::string& filename) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error opening file: " << filename << std::endl;
        return;
    }

    file << "P3\n";
    file << fb.resolution.x << " " << fb.resolution.y << "\n";
    file << "255\n";

    for (unsigned int j = 0; j < fb.resolution.y; ++j) {
        for (unsigned int i = 0; i < fb.resolution.x; ++i) {
            auto color = fb.color_buffer(i, j);
            int r = static_cast<int>(color.x * 255);
            int g = static_cast<int>(color.y * 255);
            int b = static_cast<int>(color.z * 255);
            file << r << " " << g << " " << b << "\t";
        }
        file << "\n";
    }

    file.close();
}
// STB image write-based output function for PNG
#ifdef INCLUDE_STB_IMAGE_WRITE_H
void outputPNG(const framebuffer& fb, const std::string& filename) {
    int width = fb.resolution.x;
    int height = fb.resolution.y;

    // Create a vector to store pixel data in RGBA format
    std::vector<unsigned char> pixelData(width * height * 4);

    // Fill the vector with pixel data in RGBA order
    for (int j = height - 1, k = 0; j >= 0; j--) {
        for (int i = 0; i < width; i++, k += 4) {
            auto color = fb.color_buffer(i, fb.resolution.y - j - 1);
            pixelData[k + 0] = static_cast<unsigned char>(std::max(std::min(color.x, 1.0f), 0.0f) * 255); // Red
            pixelData[k + 1] = static_cast<unsigned char>(std::max(std::min(color.y, 1.0f), 0.0f) * 255); // Green
            pixelData[k + 2] = static_cast<unsigned char>(std::max(std::min(color.z, 1.0f), 0.0f) * 255); // Blue
            pixelData[k + 3] = 255; // Alpha (fully opaque)
        }
    }

    // Use STB image write to save the PNG file
    stbi_write_png(filename.c_str(), width, height, 4, pixelData.data(), width * 4);
}
#endif
#ifdef LIBPNG_IMPL
void outputPNG(const framebuffer& fb, const std::string& filename) {
    int width = fb.resolution.x;
    int height = fb.resolution.y;

    // Create a PNG image with RGBA format
    png::image<png::rgba_pixel> image(width, height);

    // Fill the image with pixel data in RGBA order
    for (int j = height - 1; j >= 0; j--) {
        for (int i = 0; i < width; i++) {
            auto color = fb.color_buffer(i, j);
            image[j][i] = png::rgba_pixel(
                static_cast<uint8_t>(std::max(std::min(color.x, 1.0f), 0.0f) * 255), // Red
                static_cast<uint8_t>(std::max(std::min(color.y, 1.0f), 0.0f) * 255), // Green
                static_cast<uint8_t>(std::max(std::min(color.z, 1.0f), 0.0f) * 255), // Blue
                255 // Alpha (fully opaque)
            );
        }
    }

    // Save the PNG image to the specified filename
    image.write(filename);
}
#endif
void outputBMP(const framebuffer& fb, const std::string& filename) {
    std::ofstream file(filename, std::ios::out | std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error opening file: " << filename << std::endl;
        return;
    }

    int width = fb.resolution.x;
    int height = fb.resolution.y;

    // BMP file header
    const char bmpHeader[] = "BM";
    int fileSize = 54 + 3 * width * height; // 54 bytes for the header
    int reserved = 0;
    int dataOffset = 54;

    file.write(bmpHeader, 2);
    file.write((char*)&fileSize, 4);
    file.write((char*)&reserved, 4);
    file.write((char*)&dataOffset, 4);

    // DIB header
    int dibHeaderSize = 40;
    int colorPlanes = 1;
    int bitsPerPixel = 24; // 8 bits per channel (RGB)
    int compression = 0;
    int imageSize = 3 * width * height;
    int horizontalResolution = 2835; // 72 DPI
    int verticalResolution = 2835; // 72 DPI

    file.write((char*)&dibHeaderSize, 4);
    file.write((char*)&width, 4);
    file.write((char*)&height, 4);
    file.write((char*)&colorPlanes, 2);
    file.write((char*)&bitsPerPixel, 2);
    file.write((char*)&compression, 4);
    file.write((char*)&imageSize, 4);
    file.write((char*)&horizontalResolution, 4);
    file.write((char*)&verticalResolution, 4);
    file.write((char*)&reserved, 4);
    file.write((char*)&reserved, 4);

    // Write pixel data in BGR order
    for (int j = height - 1; j >= 0; j--) {
        for (int i = 0; i < width; i++) {
            auto color = fb.color_buffer(i, j);
            unsigned char b = static_cast<unsigned char>(std::max(std::min(color.x, 1.0f), 0.0f) * 255);
            unsigned char g = static_cast<unsigned char>(std::max(std::min(color.y, 1.0f), 0.0f) * 255);
            unsigned char r = static_cast<unsigned char>(std::max(std::min(color.z, 1.0f), 0.0f) * 255);
            file.write((char*)&r, 1);
            file.write((char*)&g, 1);
            file.write((char*)&b, 1);
        }
    }

    file.close();
}
#endif
#endif
