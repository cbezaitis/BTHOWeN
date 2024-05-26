# step#0: Define output directory location.
set top_module_name device_interface
set outputDir ./synth_vivado_tmp
file mkdir $outputDir
# set_part xc7vx690tffg1761-2

# Working small card
# set_part xc7k70tfbg676-3  

# Alveo big card
set_part XCU200-FSGD2104-2-E 
# set part xcu280-fsvh2892-2L-e
# set board_part xilinx.com:au280:part0:1.2


# step#1: Setup design sources and constraints.


# Put "-sv" for sustem verilog
# Put glob to catch all
read_verilog -sv [ glob ./sv_srcs/*.sv ]
#generate_target {Synthesis} [get_files ./ip/pll/pll.xci]
#read_verilog [ glob ./ip/pll/*.v ]

# step#1.1: Contraints
# Ultrascale(?)
# read_xdc ./constraints/pin.xdc

# Alveo Board
read_xdc ./constraints/alveo-u250-xdc.xdc

# step#2: Run synthesis, report utilization and timing estimates, write checkpoint design.
#synth_ip [get_ips pll] -force
set_param synth.elaboration.rodinMoreOptions "rt::set_parameter var_size_limit 7864320"
synth_design -top $top_module_name    
write_checkpoint -force $outputDir/post_synth
report_timing_summary -file $outputDir/post_synth_timing_summary.rpt
report_power -file $outputDir/post_synth_power.rpt
report_clock_interaction -delay_type min_max -file $outputDir/post_synth_clock_interaction.rpt
report_high_fanout_nets -fanout_greater_than 200 -max_nets 50 -file $outputDir/post_synth_high_fanout_nets.rpt

# step#3: Run placement and logic optimization, report utilization and timing estimates, write checkpoint design.

opt_design
place_design
phys_opt_design
write_checkpoint -force $outputDir/post_place
report_timing_summary -file $outputDir/post_place_timing_summary.rpt

# step#4: Run router, report actual utilization and timing, write checkpoint design, run drc, write verilog and xdc out.


route_design
write_checkpoint -force $outputDir/post_route
report_timing_summary -file $outputDir/post_route_timing_summary.rpt
report_timing -max_paths 100 -path_type summary -slack_lesser_than 0 -file $outputDir/post_route_setup_timing_violations.rpt
report_clock_utilization -file $outputDir/clock_util.rpt
report_utilization -file $outputDir/post_route_util.rpt
report_power -file $outputDir/post_route_power.rpt
report_drc -file $outputDir/post_imp_drc.rpt
#write_verilog -force $outputDir/top_impl_netlist.v
#write_xdc -no_fixed_only -force $outputDir/top_impl.xdc
set_property SEVERITY {Warning} [get_drc_checks NSTD-1]
set_property SEVERITY {Warning} [get_drc_checks UCIO-1]
# step#5: Generate a bitstream.
write_bitstream -force $outputDir/$top_module_name.bit


# step6
#open_hw
#open_hw_target
#current_hw_device [get_hw_devices xc7z020_1]
#refresh_hw_device -update_hw_probes false [lindex [get_hw_devices xc7z020_1] 0]

#set_property PROBES.FILE {} [get_hw_devices xc7z020_1]
#set_property FULL_PROBES.FILE {} [get_hw_devices xc7z020_1]

#set_property PROGRAM.FILE {$outputDir/top.bit} [get_hw_devices xc7z020_1]
#program_hw_devices [get_hw_devices xc7z020_1]
