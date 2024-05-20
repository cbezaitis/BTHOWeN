# Contraints
1. Put the accelerator card 
2. Put a clock on the design and contraint it for example (line 185 currently)
```
set_property PACKAGE_PIN AM10  [get_ports CLK ]; # Bank 226 Net "PEX_REFCLK_C_N" - MGTREFCLK0N_226
create_clock -period 10 -name clk [get_ports clk]
```
3. Try less period for faster results.
4. There are not enough I/O pins
5. Go the puthon file  rtl/build/make_rtl.py put less input bits. at line 39 this should write the rtl/build/global_parameters.py correctly.