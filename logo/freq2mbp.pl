#!/usr/bin/perl
use strict;
use warnings;

# Define the mapping of single-letter amino acids to their middle nucleotide base
# Based on the Standard Genetic Code
my %aa_to_base = (
    # T middle
    'F' => 'T', 'L' => 'T', 'I' => 'T', 'M' => 'T', 'V' => 'T',
    
    # C middle (S is handled separately)
    'P' => 'C', 'T' => 'C', 'A' => 'C',
    
    # A middle
    'Y' => 'A', 'H' => 'A', 'Q' => 'A', 'N' => 'A', 'K' => 'A', 'D' => 'A', 'E' => 'A',
    
    # G middle (S is handled separately)
    'C' => 'G', 'W' => 'G', 'R' => 'G', 'G' => 'G',
    
    # Optional: Stop codons if represented as '*'
    '*' => 'A', 
);

# Check for input file
die "Usage: $0 <input_tab_file.txt>\n" unless @ARGV;

open(my $fh, '<', $ARGV[0]) or die "Could not open file '$ARGV[0]': $!";

# 1. Process Header
my $header = <$fh>;
chomp $header;
my @cols = split(/\t/, $header);

# Store column indices for each base to speed up processing
my %index_map;
for (my $i = 0; $i < @cols; $i++) {
    my $aa = uc($cols[$i]);
    if ($aa eq 'S') {
        push @{$index_map{'S'}}, $i;
    } elsif (exists $aa_to_base{$aa}) {
        push @{$index_map{$aa_to_base{$aa}}}, $i;
    }
}

# 2. Print Output Header
print join("\t", "A", "C", "G", "T"), "\n";

# 3. Process Data Lines
while (my $line = <$fh>) {
    chomp $line;
    next if $line =~ /^\s*$/;
    my @val = split(/\t/, $line);
    
    my $count_A = 0;
    my $count_C = 0;
    my $count_G = 0;
    my $count_T = 0;

    # Sum standard mappings
    if ($index_map{'A'}) { $count_A += ($val[$_] // 0) for @{$index_map{'A'}}; }
    if ($index_map{'C'}) { $count_C += ($val[$_] // 0) for @{$index_map{'C'}}; }
    if ($index_map{'G'}) { $count_G += ($val[$_] // 0) for @{$index_map{'G'}}; }
    if ($index_map{'T'}) { $count_T += ($val[$_] // 0) for @{$index_map{'T'}}; }

    # Handle Serine Split (2/3 to C, 1/3 to G)
    if ($index_map{'S'}) {
        foreach my $idx (@{$index_map{'S'}}) {
            my $s_val = $val[$idx] // 0;
            $count_C += ($s_val * 2 / 3);
            $count_G += ($s_val * 1 / 3);
        }
    }

    # Print results (formatted to 2 decimal places to account for the split)
    printf("%.2f\t%.2f\t%.2f\t%.2f\n", $count_A, $count_C, $count_G, $count_T);
}

close($fh);