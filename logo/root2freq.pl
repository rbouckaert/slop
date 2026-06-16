#!/usr/bin/perl
use strict;
use warnings;

# Check for correct number of arguments
if (@ARGV < 2) {
    die "Usage: $0 <input_file> <output_file>\n";
}

my ($input_file, $output_file) = @ARGV;

# Amino acids in alphabetical order (A=0, C=1, ..., Y=19)
my @amino_acids = qw(A C D E F G H I K L M N P Q R S T V W Y);

open(my $in, '<', $input_file) or die "Could not open input file: $!\n";
open(my $out, '>', $output_file) or die "Could not open output file: $!\n";

# Read header from input and discard it
my $header = <$in>;

# Data structure to store counts: $counts[column_index][amino_acid_index]
my @counts;
my $row_count = 0;
my $num_data_cols = 0;

while (my $line = <$in>) {
    chomp $line;
    next if $line =~ /^\s*$/; # Skip empty lines
    
    my @cols = split(/\t/, $line);
    
    # Remove the first column (state number)
    shift @cols;
    
    # Track the number of data columns based on the first data row
    if ($row_count == 0) {
        $num_data_cols = scalar @cols;
    }
    
    for my $i (0 .. $#cols) {
        my $aa_index = $cols[$i];
        
        # Ensure the value is within the valid amino acid range 0-19
        if (defined $aa_index && $aa_index >= 0 && $aa_index <= 19) {
            $counts[$i][$aa_index]++;
        }
    }
    $row_count++;
}

close($in);

# Check if we actually processed any data
if ($row_count == 0) {
    die "No data rows found in input file.\n";
}

# 1. Write the output header (the Amino Acid names)
print $out join("\t", @amino_acids) . "\n";

# 2. For each column from the input file, output a line of frequencies
for my $i (0 .. $num_data_cols - 1) {
    my @freqs;
    
    for my $aa_idx (0 .. 19) {
        # Calculate frequency (count / total rows)
        my $count = $counts[$i][$aa_idx] || 0;
        my $freq = $count / $row_count;
        
        # Format to 4 decimal places for cleanliness
        push @freqs, sprintf("%d", $count);
    }
    
    print $out join("\t", @freqs) . "\n";
}

close($out);

print "Frequency table generated: $output_file\n";