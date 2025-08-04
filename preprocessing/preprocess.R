# R Script for fractional counting of multi mapped reads across multiple bam files

if (!requireNamespace("BiocManager", quietly = TRUE))
    install.packages("BiocManager")

library(GenomicFeatures)
library(rtracklayer)
library(Rsamtools)
library(GenomicAlignments)
library(GenomicRanges)

#setting workling directionary
setwd("F:\\Masterthesis") 

#loading genomic data
gff3_file <- "genomic.gff"

#creation of txdb object from the genomic data annotation
txdb <- makeTxDbFromGFF(gff3_file, format = "gff3")

#extraction of exonic ranges grouped by gene
exonsByGene <- exonsBy(txdb, by = "gene")
exons_gr <- unlist(exonsByGene)
gene_ids <- names(exons_gr)

#display of sequence levels (chromosome names) from the annotation
cat("Sequence levels in annotation (before harmonization):\n")
print(seqlevels(exons_gr))

#listing sorted BAM files
bam_files <- list.files(pattern = "_sorted\\.bam$")
if (length(bam_files) == 0) {
    stop("No BAM files found in current working directory!")
}

#reading first bam file to inspect sequence levels
first_bam <- bam_files[1]
param_diag <- ScanBamParam(
  flag = scanBamFlag(isUnmappedQuery = FALSE),
  what = "rname"
)
gal_diag <- readGAlignments(first_bam, param = param_diag)
gr_diag <- granges(gal_diag)
cat("Sequence levels in the first BAM file (", first_bam, "):\n", sep = "")
print(seqlevels(gr_diag))

# check common seq levels
common_levels <- intersect(seqlevels(exons_gr), seqlevels(gr_diag))
if (length(common_levels) == 0) {
  new_levels <- gsub("^NZ_", "", seqlevels(exons_gr))
  seqlevels(exons_gr) <- new_levels
  cat("Updated annotation seqlevels after removing 'NZ_' prefix:\n")
  print(seqlevels(exons_gr))
  common_levels <- intersect(seqlevels(exons_gr), seqlevels(gr_diag))
  if (length(common_levels) == 0) {
    warning("No common sequence levels found between annotation and BAM files. Please check your files!")
  }
} else {
  cat("Common sequence levels found between annotation and BAM files.\n")
}

#list all files ending in "_sorted.bam" (we already stored bam_files above)
cat("Found the following BAM files in current directory:\n")
print(bam_files)

#definign function to process and fractionally count single bam file
countFracBam <- function(bam_file, exons_gr, gene_ids, min_mapq = 20) {
  #indexing the BAM file if an index is not present
  if (!file.exists(paste0(bam_file, ".bai"))) {
    message("Indexing: ", bam_file)
    indexBam(bam_file)
  }
  
  #setting scan parameters: exclude unmapped reads; capture MAPQ and "NH" tag
  param <- ScanBamParam(
    flag = scanBamFlag(isUnmappedQuery = FALSE),
    what = "mapq",
    tag = "NH"
  )
  
  #reading alignments
  gal <- readGAlignments(bam_file, param = param)
  #filtering by mapping quality threshold
  gal <- gal[mcols(gal)$mapq >= min_mapq]
  gr <- granges(gal)
  
  #retrieval of multi-mapping "NH" values, assume NH=1 if missing
  nh_values <- mcols(gal)$NH
  nh_values[is.na(nh_values)] <- 1
  frac_counts <- 1 / nh_values
  
  #finding overlaps between reads and exonic regions
  overlap_hits <- findOverlaps(gr, exons_gr)
  hits_df <- as.data.frame(overlap_hits)
  hits_df$gene <- gene_ids[hits_df$subjectHits]
  hits_df$frac <- frac_counts[hits_df$queryHits]
  
  #aggregation of fractional counts per gene
  gene_counts <- aggregate(frac ~ gene, data = hits_df, FUN = sum)
  out_vec <- setNames(gene_counts$frac, gene_counts$gene)
  return(out_vec)
}

# processing all bam files and merge results to count matrix
all_results <- lapply(bam_files, function(bf) {
  message("Processing: ", bf)
  countFracBam(
    bam_file = bf,
    exons_gr = exons_gr,
    gene_ids = gene_ids,
    min_mapq = 20
  )
})

#creation of unified set of all gene ids across samples
all_genes <- unique(unlist(lapply(all_results, names)))
count_mat <- matrix(0, nrow = length(all_genes), ncol = length(all_results))
rownames(count_mat) <- all_genes
colnames(count_mat) <- gsub("_sorted\\.bam$", "", bam_files)

#filling in the count matrix from each sample result
for (i in seq_along(all_results)) {
  this_vec <- all_results[[i]]
  count_mat[names(this_vec), i] <- this_vec
}
count_df <- as.data.frame(count_mat)

#count matrix export into csv file
write.csv(count_df, "all_samples_fractional_counts.csv", row.names = TRUE)
message("Output written to all_samples_fractional_counts.csv")
print(head(count_df))
