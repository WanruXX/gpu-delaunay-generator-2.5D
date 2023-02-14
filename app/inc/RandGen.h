#ifndef GDEL2D_RANDGEN_H
#define GDEL2D_RANDGEN_H

class RandGen
{
  public:
    RandGen() = default;
    void          init(int, double, double);
    double        getNext();
    void          nextGaussian(double &, double &);
    unsigned long rand_int();

  private:
    unsigned long _z     = 0;
    unsigned long _w     = 0;
    unsigned long _jsr   = 0;
    unsigned long _jcong = 0;
    double        _min   = 0;
    double        _max   = 0;

    unsigned long znew();
    unsigned long wnew();
    unsigned long MWC();
    unsigned long SHR3();
    unsigned long CONG();
    double        random();
};

#endif //GDEL2D_RANDGEN_H