#!/usr/bin/perl
use strict;
use warnings;

# --- Configuration ---
my $char_width   = 50;
my $max_height   = 400; # Pixels for maximum bits
my $max_bits     = log(20) / log(2); # ~4.3219

# Calibration: In most Sans-Serif fonts, Uppercase letters (Cap Height) 
# are approx 71.5% of the total font-size.
my $CAP_HEIGHT_RATIO = 0.715; 

my %colors0 = (
    'G'=>'#FFA500', 'S'=>'#FFA500', 'T'=>'#FFA500', 'Y'=>'#FFC0CB', 'C'=>'#FFFF00',
    'Q'=>'#008000', 'N'=>'#008000',
    'K'=>'#0000FF', 'R'=>'#0000FF', 'H'=>'#0000FF',
    'D'=>'#FF0000', 'E'=>'#FF0000',
    'A'=>'#FFA500', 'V'=>'#000000', 'L'=>'#000000', 'I'=>'#000000',
    'P'=>'#000000', 'W'=>'#000000', 'F'=>'#000000', 'M'=>'#000000'
);

my %colors = (
		'A'=> '#E6E600', 'V'=> '#E6E600', 'L'=> '#E6E600', 'I'=> '#E6E600', 'M'=> '#E6E600', 
        'F'=> '#32CD32', 'W'=> '#32CD32', 'Y'=> '#32CD32',
        'K'=> '#0000FF', 'R'=> '#0000FF', 'H'=> '#0000FF', 
        'D'=> '#FF0000', 'E'=> '#FF0000',
        'G'=> '#888888', 'P'=> '#888888',
        'S'=> '#FF8C00', 'T'=> '#FF8C00', 'N'=> '#FF8C00', 'Q'=> '#FF8C00', 'C'=> '#FF8C00'
);


# --- 1. Parse Input ---
# Check for correct number of arguments
if (@ARGV < 2) {
    die "Usage: $0 <input.freq> <output.svg>\n";
}

my ($input, $output_file) = @ARGV;
open(my $fh, '<', $input) or die "Can't open $input: $!\n";

my $header = <$fh>;
chomp $header;
my @aa_names = split(/\t/, $header);

my @site_data;
while (<$fh>) {
    chomp;
    next if /^\s*$/;
    push @site_data, [ split(/\t/) ];
}
close($fh);

# --- 2. SVG Setup ---
my $pad = 60;
my $svg_w = (@site_data * $char_width) + ($pad * 2);
my $svg_h = $max_height + ($pad * 2);

open(my $out, '>', $output_file) or die "Can't write $output_file: $!\n";

print $out qq{<?xml version="1.0" encoding="UTF-8" standalone="no"?>
<svg width="$svg_w" height="$svg_h" viewBox="0 0 $svg_w $svg_h" xmlns="http://www.w3.org/2000/svg">
<rect width="100%" height="100%" fill="white" />
<line x1="$pad" y1="$pad" x2="$pad" y2="@{[$max_height+$pad]}" stroke="black" stroke-width="1.5" />
<line x1="$pad" y1="@{[$max_height+$pad]}" x2="@{[$svg_w-$pad]}" y2="@{[$max_height+$pad]}" stroke="black" stroke-width="1.5" />
};

# --- 3. Logic & Drawing ---
for my $i (0 .. $#site_data) {
    my @counts = @{$site_data[$i]};
    my $total_counts = 0.0;
    $total_counts += $_ for @counts;
    next if $total_counts == 0.0;

    # Information Content (R)
    my $entropy = 0.0;
    my @probs;
    for my $j (0 .. $#counts) {
        my $p = $counts[$j] / $total_counts;
        if ($p > 0.0) {
            $entropy -= $p * (log($p) / log(2));
        }
        push @probs, { char => $aa_names[$j], p => $p };
    }
    
    my $ic = $max_bits;# - $entropy;
    # Small sample correction could go here, but we'll assume large N for simplicity
    $ic = 0 if $ic < 0;

    # Sort: Smallest probability first (draws bottom-up)
    @probs = sort { $a->{p} <=> $b->{p} } @probs;

    my $x_center = $pad + ($i * $char_width) + ($char_width / 2);
    my $current_y_baseline = $max_height + $pad;

    foreach my $item (@probs) {
        next if $item->{p} <= 0;
        
        # Calculate height in pixels
        # Height = (Prob * IC / MaxBits) * TotalPixelHeight
        my $h = ($item->{p} * $ic / $max_bits) * $max_height;
        
        # Skip if height is effectively sub-pixel
        next if $h < 0.1;

        my $color = $colors{$item->{char}} || "#777777";
        
        # FONT SCALING LOGIC:
        # We want the uppercase letter to be exactly $h pixels tall.
        # If font-size is 100, the letter is (100 * $CAP_HEIGHT_RATIO) pixels tall.
        my $nominal_size = 100.0;
        my $real_cap_height = $nominal_size * $CAP_HEIGHT_RATIO;
        my $v_scale = $h / $real_cap_height;
        
        # Horizontal scale: stretch to 90% of column width
        my $h_scale = ($char_width * 0.9) / ($nominal_size * 0.6); # 0.6 is approx width/height ratio

        # Use printf with %.6f to avoid integer rounding gaps
        printf $out qq{  <g transform="translate(%.6f, %.6f) scale(%.6f, %.6f)">\n}, 
               $x_center, $current_y_baseline, $h_scale, $v_scale;
        printf $out qq{    <text x="0" y="0" font-family="Arial, Helvetica, sans-serif" font-weight="bold" font-size="%.1f" fill="%s" text-anchor="middle">%.1s</text>\n}, 
               $nominal_size, $color, $item->{char};
        print $out qq{  </g>\n};

        # Subtract exactly what we added to the stack
        $current_y_baseline -= $h;
    }

    # X-Axis index
    print $out qq{<text x="$x_center" y="@{[$max_height + $pad + 20]}" font-family="Arial" font-size="12" text-anchor="middle">@{[$i+1]}</text>\n};
}

# --- 4. Y-Axis Labels (Bits) ---
for my $b (0 .. 4) {
    my $y_val = ($max_height + $pad) - ($b / $max_bits * $max_height);
    printf $out qq{<line x1="%d" y1="%.4f" x2="%d" y2="%.4f" stroke="black" />\n}, $pad-5, $y_val, $pad, $y_val;
    printf $out qq{<text x="%d" y="%.4f" font-family="Arial" font-size="12" text-anchor="end" alignment-baseline="middle">%s</text>\n}, $pad-10, $y_val, $b;
}

print $out "</svg>\n";
close($out);
print "Logo created: $output_file\n";