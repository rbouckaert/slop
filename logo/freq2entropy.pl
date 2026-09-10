$s= <>;
print $s;
while ($s = <>) {
	@s = split('\s', $s);
	$sum = 0.0;
	foreach $i (@s) {
		$sum += $i;
	}
	$h = 0.0;
	foreach $i (@s) {
		if ($i > 0) {
			$p = $i/$sum;
			$h +=  -$p * log($p);
		}
	}
	print "$h\n";
}
