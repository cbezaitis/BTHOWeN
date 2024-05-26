# Contraints
1. Put the accelerator card 
2. Put a clock on the design and contraint it for example (line 185 currently)
```
set_property PACKAGE_PIN AM10  [get_ports CLK ]; # Bank 226 Net "PEX_REFCLK_C_N" - MGTREFCLK0N_226
create_clock -period 10 -name clk [get_ports clk]
```
3. Try less period for faster results.
4. In the tcl use as top the __device_interface__ System Verilog Module
5. Specify the limit size of parameters is important with 
```
set_param synth.elaboration.rodinMoreOptions "rt::set_parameter var_size_limit 7864320"
```
6. Then, the TCL script can be executed correctly
