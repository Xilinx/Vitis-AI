## if any bitfile apart from _partial bitfile exists (top level bitfile), then delete partial bitfiles
# if { [lsearch -not [glob -nocomplain *.bit] "*_partial.bit"] >= 0  || [lsearch -not [glob -nocomplain *.bit] "*_partial_clear.bit"] >= 0 } {
# 	catch {file delete -force {*}[glob -nocomplain *_partial.bit]};
# 	catch {file delete -force {*}[glob -nocomplain *_partial_clear.bit]};
# }
if { [lsearch -not -regexp [glob -nocomplain *.bit] ".*_partial(_clear)*.bit"] >= 0 } {
	catch {file delete -force {*}[glob -nocomplain *_partial.bit]};
	catch {file delete -force {*}[glob -nocomplain *_partial_clear.bit]};
}
