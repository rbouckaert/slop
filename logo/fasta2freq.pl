#!/usr/bin/perl
use strict;
use warnings;

# Check for correct number of arguments
if (@ARGV < 2) {
    die "Usage: $0 <input.fasta> <output_file>\n";
}

my ($input_file, $output_file) = @ARGV;

# Amino acids in alphabetical order
my @aa_list = qw(A C D E F G H I K L M N P Q R S T V W Y);

# Create a lookup hash for quick indexing (A=0, C=1, etc.)
my %aa_to_idx;
for my $i (0 .. $#aa_list) {
    $aa_to_idx{$aa_list[$i]} = $i;
}

# 1. Parse FASTA file
my @sequences;
my $current_seq = "";

open(my $in, '<', $input_file) or die "Could not open $input_file: $!\n";
while (my $line = <$in>) {
    chomp $line;
    if ($line =~ /^>/) {
        # Save previous sequence if it exists
        push @sequences, $current_seq if $current_seq ne "";
        $current_seq = "";
    } else {
        # Support multi-line fasta sequences
        $line =~ s/\s+//g; # Remove any internal whitespace
        $current_seq .= uc($line); # Convert to uppercase
    }
}
push @sequences, $current_seq if $current_seq ne ""; # Add the last one
close($in);

if (!@sequences) {
    die "Error: No sequences found in the input file.\n";
}

# Find the length of the longest sequence (alignment length)
my $max_len = 0;
foreach my $seq (@sequences) {
    my $len = length($seq);
    $max_len = $len if $len > $max_len;
}

# 2. Count frequencies per column
# $counts[column_index][aa_index]
my @counts;

foreach my $seq (@sequences) {
    my @chars = split('', $seq);
    for my $i (0 .. $#chars) {
        my $char = $chars[$i];
        if (exists $aa_to_idx{$char}) {
            $counts[$i][$aa_to_idx{$char}]++;
        }
    }
}

# 3. Output the results
open(my $out, '>', $output_file) or die "Could not open $output_file: $!\n";

# Print header
print $out join("\t", @aa_list) . "\n";

# Print frequencies for each column
for my $i (0 .. $max_len - 1) {
    my @row_freqs;
    my $total_sequences = scalar @sequences;
    
    for my $idx (0 .. 19) {
        my $count = $counts[$i][$idx] || 0;
        my $freq = $count / $total_sequences;
        push @row_freqs, sprintf("%d", $count);
    }
    print $out join("\t", @row_freqs) . "\n";
}

close($out);
print "Processed " . scalar(@sequences) . " sequences.\n";
print "Frequency table saved to: $output_file\n";