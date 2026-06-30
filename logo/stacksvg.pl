#!/usr/bin/perl
use strict;
use warnings;

# --- Configuration ---
my $FONT_SIZE   = 36;   # Font size in pixels
my $TEXT_MARGIN = 5;    # Space between text and the image below it
my $BLOCK_GAP   = 20;   # Space between the end of one image and the start of the next label
my $LABEL_SPACE = $FONT_SIZE + $TEXT_MARGIN;

if (@ARGV < 2) {
    die "Usage: $0 file1.svg file2.svg [file3.svg ...] > output.svg\n";
}

my $total_height = 0;
my $max_width    = 0;
my @svg_data;

foreach my $file (@ARGV) {
    open(my $fh, '<', $file) or die "Could not open $file: $!\n";
    my $content = do { local $/; <$fh> };
    close($fh);

    # 1. Generate Label (Remove path and .svg extension)
    my $label = $file;
    $label =~ s/.*\///;      # Remove directory path (e.g., "dir/file.svg" -> "file.svg")
    $label =~ s/\.svg$//i;   # Remove extension (e.g., "file.svg" -> "file")

    # 2. Extract dimensions from <svg> tag
    if ($content =~ /<svg([^>]+)>/is) {
        my $attr_string = $1;
        
        my ($w) = $attr_string =~ /width\s*=\s*["']([\d.]+)/i;
        my ($h) = $attr_string =~ /height\s*=\s*["']([\d.]+)/i;

        # Fallback to viewBox
        if (!$w || !$h) {
            if ($attr_string =~ /viewBox\s*=\s*["'][\d.]+\s+[\d.]+\s+([\d.]+)\s+([\d.]+)["']/i) {
                $w ||= $1; $h ||= $2;
            }
        }
        $w ||= 300; $h ||= 150; # Defaults if nothing found

        # 3. Extract internal content
        my $inner = $content;
        $inner =~ s/.*?<svg[^>]*>//is;
        $inner =~ s/<\/svg>.*//is;

        push @svg_data, {
            body   => $inner,
            width  => $w,
            height => $h,
            label  => $label
        };

        # Add up the height: (Label + Margin + Image + Gap)
        $total_height += ($LABEL_SPACE + $h + $BLOCK_GAP);
        $max_width = $w if $w > $max_width;
    }
}

# 4. Output the combined SVG
print qq|<?xml version="1.0" encoding="UTF-8" standalone="no"?>\n|;
print qq|<svg width="$max_width" height="$total_height" xmlns="http://www.w3.org/2000/svg">\n|;
print qq|  <style> .label-style { font: bold ${FONT_SIZE}px sans-serif; fill: #333; } </style>\n|;

my $current_y = 0;
foreach my $item (@svg_data) {
    # Calculate position for the text baseline
    my $text_baseline = $current_y + $FONT_SIZE;
    
    # Calculate position for the SVG group
    my $svg_top = $current_y + $LABEL_SPACE;

    print qq|  <!-- Group for $item->{label} -->\n|;
    
    # Print the filename label
    print qq|  <text x="0" y="$text_baseline" class="label-style">$item->{label}</text>\n|;
    
    # Print the SVG content wrapped in a translated group
    print qq|  <g transform="translate(0, $svg_top)">\n|;
    print $item->{body};
    print qq|  </g>\n|;

    # Advance current_y for the next block
    $current_y += ($LABEL_SPACE + $item->{height} + $BLOCK_GAP);
}

print qq|</svg>\n|;