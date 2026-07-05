#!/usr/bin/perl
use strict;
use warnings;

# Usage: perl convert_version.pl version.xml > module-info.java
my $xml_file = $ARGV[0] || 'version.xml';

if (!-e $xml_file) {
    die "Error: Cannot find file '$xml_file'\nUsage: $0 <version.xml>\n";
}

# Slurp the whole file into a string
my $xml_content;
{
    local $/;
    open my $fh, '<', $xml_file or die "Could not open $xml_file: $!";
    $xml_content = <$fh>;
    close $fh;
}

# 1. Extract Module Name from <package name='...'>
my $module_name = "UNKNOWN";
if ($xml_content =~ m|<package\s+[^>]*name=['"]([^'"]+)['"]|i) {
    $module_name = $1;
}

# Data structures to hold findings
my %packages;
my @services;

# 2. Extract Services and Providers
# Matches <service type="..."> ... </service>
while ($xml_content =~ m|<service\s+type="([^"]+)"\s*>(.*?)</service>|sg) {
    my $service_type = $1;
    my $service_body = $2;
    my @providers;

    # Inside the service body, find all <provider classname="..." />
    while ($service_body =~ m|<provider\s+classname="([^"]+)"\s*/>|g) {
        my $classname = $1;
        push @providers, $classname;

        # Extract package name (everything before the last dot)
        if ($classname =~ /^(.*)\.[^.]+$/) {
            $packages{$1} = 1;
        }
    }

    if (@providers) {
        push @services, {
            type => $service_type,
            providers => \@providers
        };
    }
}

# --- Output Generation ---

print "module $module_name {\n";
print "    requires beast.base;\n";
print "    requires beast.fx;\n";
print "    requires org.apache.commons.statistics.distribution;\n\n";

# Print Exports
if (keys %packages) {
    print "    // Exports\n";
    foreach my $pkg (sort keys %packages) {
        print "    exports $pkg;\n";
    }
    print "\n";
}

# Print Provides
if (@services) {
    print "    // Services\n";
    foreach my $service (@services) {
        print "    provides " . $service->{type} . " with\n";
        my $last_idx = $#{$service->{providers}};
        for my $i (0 .. $last_idx) {
            my $sep = ($i == $last_idx) ? ";" : ",";
            print "        " . $service->{providers}->[$i] . "$sep\n";
        }
        print "\n";
    }
}

print "}\n";